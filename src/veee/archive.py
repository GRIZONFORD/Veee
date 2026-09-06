r"""
archive.py — Ingesta del archivo historico gratuito de Football-Data.co.uk.

Por que este modulo es la columna vertebral del estudio
-------------------------------------------------------
Una temporada de recoleccion prospectiva (n~760) solo detecta un ROI del 9%,
magnitud que nadie sostiene que exista en un mercado maduro. El archivo historico
—gratuito, sin clave y disponible hoy— eleva la muestra a decenas de miles de
partidos y baja el efecto minimo detectable por debajo del 1%, que es el orden de
magnitud del alfa que documenta la literatura.

Ademas resuelve el problema de identificacion del diseno original: la cuota de
**otra casa** es un conjunto de informacion genuinamente independiente del precio
sobre el que se evalua, cosa que la probabilidad neutralizada del propio libro
nunca podia ser.

Principio de diseno: DESCUBRIR, NO SUPONER
------------------------------------------
El esquema de estos CSV **no es estable entre temporadas**. Cambia de formato de
fecha, de codificacion, de nombres de columna agregada (`BbMxH` -> `MaxH`), y las
columnas de cuota de cierre no existen en las temporadas antiguas. Un
`pd.read_csv` seguido de acceso directo a columnas produciria columnas ausentes
en silencio y una muestra sesgada por omision.

Por eso el modulo nunca supone que una columna exista: **inventaria lo que hay**
(`discover_schema`) y emite un informe de cobertura por temporada, liga y casa.
Los patrones de nombre del registro provienen de la documentacion del sitio y
**deben confirmarse contra los ficheros reales**; el informe de cobertura es
precisamente el mecanismo de verificacion.

Contrato de salida
------------------
`load_season` devuelve dos tablas en formato largo, listas para la econometria:

  matches : una fila por partido (resultado, corners, tarjetas, tiros...)
  odds    : una fila por (partido, casa, mercado, fase, seleccion) -> cuota

El formato largo es deliberado: permite responder con simples agrupaciones a
"que casa bate el cierre", "cuanta dispersion hay entre casas" o "predice el
desplazamiento apertura-cierre", sin remodelar nada.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# --------------------------------------------------------------------------- #
# 1. Registros de ligas y casas                                                #
# --------------------------------------------------------------------------- #
LIGAS: dict[str, str] = {
    "SP1": "LaLiga", "SP2": "LaLiga 2",
    "E0": "Premier League", "E1": "Championship",
    "D1": "Bundesliga", "D2": "Bundesliga 2",
    "I1": "Serie A", "I2": "Serie B",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "P1": "Primeira Liga",
    "B1": "Jupiler League", "SC0": "Scottish Premiership",
    "T1": "Super Lig", "G1": "Super League Grecia",
}

# Prefijo de columna -> (nombre, es_referencia_sharp).
# Pinnacle es la referencia sharp estandar de la literatura: opera con margen
# minimo y no limita al apostador ganador, de modo que su linea de cierre es el
# mejor estimador disponible de la probabilidad real.
CASAS: dict[str, tuple[str, bool]] = {
    "B365": ("Bet365", False), "BS": ("Blue Square", False),
    "BW": ("Bet&Win", False), "GB": ("Gamebookers", False),
    "IW": ("Interwetten", False), "LB": ("Ladbrokes", False),
    "PS": ("Pinnacle", True), "SO": ("Sporting Odds", False),
    "SB": ("Sportingbet", False), "SJ": ("Stan James", False),
    "SY": ("Stanleybet", False), "VC": ("VC Bet", False),
    "WH": ("William Hill", False),
    # Agregados de mercado, no casas: se marcan aparte porque no son
    # cotizaciones ejecutables sino estadisticos de resumen.
    "Max": ("Maximo del mercado", False), "Avg": ("Media del mercado", False),
    "BbMx": ("Maximo (Betbrain)", False), "BbAv": ("Media (Betbrain)", False),
}
AGREGADOS = {"Max", "Avg", "BbMx", "BbAv"}

# Columnas de resultado y estadisticas de partido.
COLS_RESULTADO = {
    "FTHG": "goles_local", "FTAG": "goles_visitante", "FTR": "resultado_ft",
    "HTHG": "goles_local_ht", "HTAG": "goles_visitante_ht", "HTR": "resultado_ht",
}
COLS_ESTADISTICAS = {
    "HS": "tiros_local", "AS": "tiros_visitante",
    "HST": "tiros_puerta_local", "AST": "tiros_puerta_visitante",
    "HC": "corners_local", "AC": "corners_visitante",
    "HF": "faltas_local", "AF": "faltas_visitante",
    "HY": "amarillas_local", "AY": "amarillas_visitante",
    "HR": "rojas_local", "AR": "rojas_visitante",
}


# --------------------------------------------------------------------------- #
# 2. Patrones de columnas de cuota                                             #
# --------------------------------------------------------------------------- #
def cols_1x2(prefijo: str, cierre: bool = False) -> dict[str, str]:
    """Columnas 1X2 de una casa. El cierre se marca con una 'C' tras el prefijo."""
    p = f"{prefijo}C" if cierre else prefijo
    return {"home": f"{p}H", "draw": f"{p}D", "away": f"{p}A"}


def cols_ou25(prefijo: str, cierre: bool = False) -> dict[str, str]:
    """Columnas Over/Under 2.5 goles.

    Pinnacle usa el prefijo corto 'P' en estos mercados (no 'PS'), y los
    agregados antiguos llevan el prefijo 'Bb'. La irregularidad es del origen,
    no del modulo.
    """
    base = "P" if prefijo == "PS" else prefijo
    p = f"{base}C" if cierre else base
    return {"over": f"{p}>2.5", "under": f"{p}<2.5"}


# La columna que porta la LINEA del handicap cambio de nombre con el rediseno del
# archivo: los ficheros antiguos la llaman 'BbAHh' y los modernos 'AHh'/'AHCh'.
# Una cuota asiatica sin su linea es inliquidable, de modo que se prueban todas
# las variantes documentadas y se conserva la primera presente.
LINEA_AH_APERTURA = ("AHh", "BbAHh", "AHCh")
LINEA_AH_CIERRE = ("AHCh", "AHh", "BbAHh")


def cols_ah(prefijo: str, cierre: bool = False) -> dict[str, Any]:
    """Columnas de handicap asiatico, con las candidatas para la linea aplicada."""
    base = "P" if prefijo == "PS" else prefijo
    p = f"{base}C" if cierre else base
    candidatas = LINEA_AH_CIERRE if cierre else LINEA_AH_APERTURA
    return {"home": f"{p}AHH", "away": f"{p}AHA", "_linea_candidatas": candidatas}


MERCADOS = {
    "1X2": cols_1x2,
    "ou_2.5": cols_ou25,
    "ah": cols_ah,
}


# --------------------------------------------------------------------------- #
# 3. Lectura robusta del CSV                                                   #
# --------------------------------------------------------------------------- #
_CODIFICACIONES = ("utf-8", "cp1252", "latin-1")


def read_csv_robusto(origen: str | Path | bytes) -> pd.DataFrame:
    """Lee un CSV del archivo tolerando sus irregularidades conocidas.

    Tres problemas reales de estos ficheros, cada uno capaz de corromper la
    muestra en silencio:

    1. **Codificacion.** Los nombres de equipo llevan acentos y los ficheros no
       son UTF-8, sino cp1252/latin-1. Decodificar mal no lanza error: produce
       nombres corruptos que luego no cruzan entre temporadas.
    2. **Comas finales.** Generan columnas fantasma 'Unnamed: N' que rompen
       cualquier iteracion sobre columnas.
    3. **Filas en blanco al final.** Se cuelan como partidos con todo a NaN e
       inflan artificialmente el tamano muestral.
    """
    datos = origen if isinstance(origen, bytes) else Path(origen).read_bytes()
    ultimo: Exception | None = None
    for cod in _CODIFICACIONES:
        try:
            df = pd.read_csv(io.BytesIO(datos), encoding=cod, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            ultimo = exc
    else:
        raise ValueError(f"No se pudo decodificar el CSV: {ultimo}")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")
    if "HomeTeam" in df.columns:
        df = df[df["HomeTeam"].notna() & (df["HomeTeam"].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


def parse_fechas(fechas: pd.Series, horas: pd.Series | None = None) -> pd.Series:
    """Convierte la columna Date, que alterna 'dd/mm/yy' y 'dd/mm/yyyy'.

    `dayfirst=True` es obligatorio: sin el, '03/04/2024' se interpreta como 4 de
    marzo en lugar de 3 de abril. El error no lanza excepcion y desordena toda la
    serie temporal, que es justo lo que arruinaria un analisis apertura-cierre.
    """
    s = fechas.astype(str).str.strip()
    fecha = pd.to_datetime(s, format="%d/%m/%Y", dayfirst=True, errors="coerce")
    faltan = fecha.isna()
    if faltan.any():
        fecha.loc[faltan] = pd.to_datetime(s[faltan], format="%d/%m/%y",
                                           dayfirst=True, errors="coerce")
    faltan = fecha.isna()
    if faltan.any():   # ultimo recurso, con dayfirst explicito
        fecha.loc[faltan] = pd.to_datetime(s[faltan], dayfirst=True, errors="coerce")

    if horas is not None and horas.notna().any():
        h = pd.to_datetime(horas.astype(str).str.strip(), format="%H:%M",
                           errors="coerce").dt.time
        fecha = pd.Series([
            pd.Timestamp.combine(f.date(), t) if pd.notna(f) and pd.notna(t) else f
            for f, t in zip(fecha, h)
        ], index=fecha.index)
    return fecha


# --------------------------------------------------------------------------- #
# 4. Descubrimiento de esquema                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class EsquemaDescubierto:
    """Inventario de lo que un fichero concreto contiene de verdad."""
    n_filas: int
    columnas: list[str]
    tiene_resultado: bool
    estadisticas: list[str] = field(default_factory=list)
    # (casa, mercado, fase) -> {seleccion: columna}
    cuotas: dict[tuple[str, str, str], dict[str, str]] = field(default_factory=dict)

    @property
    def casas(self) -> set[str]:
        return {c for c, _, _ in self.cuotas}

    @property
    def tiene_cierre(self) -> bool:
        return any(f == "cierre" for _, _, f in self.cuotas)

    def resumen(self) -> dict[str, Any]:
        return {
            "n_filas": self.n_filas, "n_columnas": len(self.columnas),
            "tiene_resultado": self.tiene_resultado,
            "n_estadisticas": len(self.estadisticas),
            "casas": sorted(self.casas),
            "tiene_cierre": self.tiene_cierre,
            "mercados": sorted({m for _, m, _ in self.cuotas}),
        }


def discover_schema(df: pd.DataFrame) -> EsquemaDescubierto:
    """Inventaria que columnas de cuota, resultado y estadisticas existen.

    Es el corazon del modulo: en vez de suponer un esquema fijo, comprueba cuales
    de los patrones documentados estan realmente presentes. Lo que no aparezca
    quedara reflejado en el informe de cobertura en lugar de fallar en silencio.
    """
    cols = set(df.columns.astype(str))
    esquema = EsquemaDescubierto(
        n_filas=len(df),
        columnas=sorted(cols),
        tiene_resultado={"FTHG", "FTAG"} <= cols,
        estadisticas=[c for c in COLS_ESTADISTICAS if c in cols],
    )
    for prefijo in CASAS:
        for mercado, constructor in MERCADOS.items():
            for cierre, fase in ((False, "apertura"), (True, "cierre")):
                mapa = constructor(prefijo, cierre)
                presentes = {k: v for k, v in mapa.items()
                             if not k.startswith("_") and v in cols}
                # Un mercado exige al menos dos selecciones para poder neutralizar
                # el margen; con una sola la observacion es inutilizable.
                if len(presentes) < 2:
                    continue
                for cand in mapa.get("_linea_candidatas", ()):
                    if cand in cols:
                        presentes["_linea"] = cand
                        break
                esquema.cuotas[(prefijo, mercado, fase)] = presentes
    return esquema


# --------------------------------------------------------------------------- #
# 5. Normalizacion                                                             #
# --------------------------------------------------------------------------- #
def match_id(div: str, temporada: str, fecha: pd.Timestamp,
             local: str, visitante: str) -> str:
    """Identificador determinista y estable entre temporadas."""
    dia = fecha.strftime("%Y-%m-%d") if pd.notna(fecha) else "NA"
    crudo = f"{div}|{temporada}|{dia}|{str(local).strip()}|{str(visitante).strip()}"
    return hashlib.sha1(crudo.lower().encode()).hexdigest()[:16]


def _a_numero(s: pd.Series) -> pd.Series:
    """Convierte a float tolerando comas decimales y celdas vacias."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.strip()
         .replace({"": None, "nan": None, "None": None}),
        errors="coerce")


def normalizar_partidos(df: pd.DataFrame, div: str, temporada: str) -> pd.DataFrame:
    """Tabla de partidos con resultado y estadisticas observadas."""
    fecha = parse_fechas(df["Date"], df["Time"] if "Time" in df.columns else None)
    out = pd.DataFrame({
        "match_id": [match_id(div, temporada, f, h, a)
                     for f, h, a in zip(fecha, df["HomeTeam"], df["AwayTeam"])],
        "div": div, "liga": LIGAS.get(div, div), "temporada": temporada,
        "fecha": fecha,
        "equipo_local": df["HomeTeam"].astype(str).str.strip(),
        "equipo_visitante": df["AwayTeam"].astype(str).str.strip(),
    })
    for origen, destino in {**COLS_RESULTADO, **COLS_ESTADISTICAS}.items():
        if origen in df.columns:
            out[destino] = (df[origen].astype(str).str.strip()
                            if origen in ("FTR", "HTR") else _a_numero(df[origen]))
        else:
            out[destino] = np.nan

    # Variables derivadas que alimentan directamente los mercados del estudio.
    out["corners_totales"] = out["corners_local"] + out["corners_visitante"]
    out["tarjetas_totales"] = out["amarillas_local"] + out["amarillas_visitante"]
    out["goles_totales"] = out["goles_local"] + out["goles_visitante"]
    out["margen_goles"] = out["goles_local"] - out["goles_visitante"]
    return out


def normalizar_cuotas(df: pd.DataFrame, esquema: EsquemaDescubierto,
                      partidos: pd.DataFrame) -> pd.DataFrame:
    """Aplana a formato largo: una fila por (partido, casa, mercado, fase, seleccion).

    El formato largo permite responder con agrupaciones simples a las preguntas
    del estudio (dispersion entre casas, CLV por casa, desplazamiento
    apertura-cierre) sin tener que remodelar la tabla en cada analisis.
    """
    filas: list[pd.DataFrame] = []
    for (prefijo, mercado, fase), mapa in esquema.cuotas.items():
        linea_col = mapa.get("_linea")
        linea = _a_numero(df[linea_col]) if linea_col else pd.Series(np.nan, index=df.index)
        for seleccion, col in mapa.items():
            if seleccion.startswith("_"):
                continue
            cuota = _a_numero(df[col])
            valido = cuota.notna() & (cuota >= 1.01) & (cuota <= 1001.0)
            if not valido.any():
                continue
            filas.append(pd.DataFrame({
                "match_id": partidos["match_id"][valido].values,
                "casa": CASAS.get(prefijo, (prefijo, False))[0],
                "prefijo": prefijo,
                "es_sharp": CASAS.get(prefijo, (prefijo, False))[1],
                "es_agregado": prefijo in AGREGADOS,
                "mercado": mercado, "fase": fase, "seleccion": seleccion,
                "linea": linea[valido].values,
                "cuota": cuota[valido].values,
                "columna_origen": col,
            }))
    if not filas:
        return pd.DataFrame(columns=["match_id", "casa", "prefijo", "es_sharp",
                                     "es_agregado", "mercado", "fase", "seleccion",
                                     "linea", "cuota", "columna_origen"])
    return pd.concat(filas, ignore_index=True)


def validar(partidos: pd.DataFrame, cuotas: pd.DataFrame) -> list[str]:
    """Comprobaciones de integridad. Devuelve la lista de anomalias detectadas.

    Ninguna de estas condiciones lanza excepcion: la muestra sigue siendo
    utilizable si se declara el problema. Lo inaceptable seria que pasaran
    inadvertidas y contaminaran el analisis.
    """
    avisos: list[str] = []
    if partidos["fecha"].isna().any():
        avisos.append(f"{int(partidos['fecha'].isna().sum())} partidos sin fecha "
                      "interpretable (revisar el formato de Date).")
    dup = partidos["match_id"].duplicated().sum()
    if dup:
        avisos.append(f"{int(dup)} match_id duplicados: posible doble jornada "
                      "el mismo dia entre los mismos equipos.")
    ah = cuotas[cuotas["mercado"] == "ah"]
    if not ah.empty and ah["linea"].isna().any():
        n = int(ah["linea"].isna().sum())
        avisos.append(f"{n} cuotas asiaticas sin linea: INLIQUIDABLES, se deben "
                      "excluir del analisis de handicap.")
    if not partidos["goles_local"].notna().any():
        avisos.append("Ningun partido con resultado: no se podra liquidar nada.")
    # Un libro cuya suma de probabilidades implicitas es < 1 seria un arbitraje
    # puro; en la practica casi siempre indica un error de columna o de lectura.
    for (mid, casa, mercado, fase), g in cuotas.groupby(
            ["match_id", "casa", "mercado", "fase"], observed=True):
        if len(g) >= 2:
            suma = float((1.0 / g["cuota"]).sum())
            if suma < 0.98:
                avisos.append(f"Libro con suma implicita {suma:.4f} < 1 en "
                              f"{casa}/{mercado}/{fase}: revisar mapeo de columnas.")
                break
    return avisos


@dataclass
class TemporadaCargada:
    div: str
    temporada: str
    partidos: pd.DataFrame
    cuotas: pd.DataFrame
    esquema: EsquemaDescubierto
    avisos: list[str] = field(default_factory=list)


def load_season(origen: str | Path | bytes, div: str,
                temporada: str) -> TemporadaCargada:
    """Carga un CSV de temporada y lo normaliza sin suponer esquema."""
    df = read_csv_robusto(origen)
    faltan = {"Date", "HomeTeam", "AwayTeam"} - set(df.columns)
    if faltan:
        raise ValueError(f"{div} {temporada}: faltan columnas esenciales {sorted(faltan)}. "
                         "¿Es un fichero del archivo principal de Football-Data?")
    esquema = discover_schema(df)
    partidos = normalizar_partidos(df, div, temporada)
    cuotas = normalizar_cuotas(df, esquema, partidos)
    avisos = validar(partidos, cuotas)
    log.info("%s %s: %d partidos, %d cotizaciones, %d casas, cierre=%s",
             div, temporada, len(partidos), len(cuotas),
             len(esquema.casas), esquema.tiene_cierre)
    for a in avisos:
        log.warning("%s %s: %s", div, temporada, a)
    return TemporadaCargada(div, temporada, partidos, cuotas, esquema, avisos)


# --------------------------------------------------------------------------- #
# 6. Descarga y cache                                                          #
# --------------------------------------------------------------------------- #
def codigo_temporada(anio_inicio: int) -> str:
    """2024 -> '2425' (formato de URL del sitio)."""
    return f"{anio_inicio % 100:02d}{(anio_inicio + 1) % 100:02d}"


def etiqueta_temporada(anio_inicio: int) -> str:
    """2024 -> '2024-25'."""
    return f"{anio_inicio}-{(anio_inicio + 1) % 100:02d}"


def url_temporada(div: str, anio_inicio: int) -> str:
    return f"{BASE_URL}/{codigo_temporada(anio_inicio)}/{div}.csv"


def descargar(div: str, anio_inicio: int, cache_dir: str | Path = "data/archive",
              session: Any = None, forzar: bool = False) -> Path:
    """Descarga un CSV al cache local. No vuelve a pedirlo si ya existe.

    El archivo historico es inmutable salvo para la temporada en curso, de modo
    que cachear es correcto ademas de cortes con el servidor.
    """
    destino = Path(cache_dir) / codigo_temporada(anio_inicio) / f"{div}.csv"
    if destino.exists() and not forzar:
        log.debug("Cache: %s", destino)
        return destino
    from .scrapers.base import PoliteSession
    session = session or PoliteSession(rate_limit_s=1.5, cache_dir=cache_dir)
    r = session.get(url_temporada(div, anio_inicio), cache=False)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    log.info("Descargado %s (%d bytes)", destino, len(r.content))
    return destino


def load_many(divs: Iterable[str], anios: Iterable[int],
              cache_dir: str | Path = "data/archive",
              descargar_si_falta: bool = True) -> list[TemporadaCargada]:
    """Carga varias ligas y temporadas, omitiendo con aviso las no disponibles.

    Que falte una combinacion liga-temporada es normal (ascensos, cambios de
    cobertura del sitio). Se registra y se continua: abortar todo el barrido por
    un fichero ausente seria peor.
    """
    salida: list[TemporadaCargada] = []
    for anio in anios:
        for div in divs:
            ruta = Path(cache_dir) / codigo_temporada(anio) / f"{div}.csv"
            try:
                if not ruta.exists():
                    if not descargar_si_falta:
                        log.warning("Ausente y sin descarga: %s %s", div, anio)
                        continue
                    ruta = descargar(div, anio, cache_dir)
                salida.append(load_season(ruta, div, etiqueta_temporada(anio)))
            except Exception as exc:                       # noqa: BLE001
                log.warning("Omitido %s %s: %s", div, etiqueta_temporada(anio), exc)
    return salida


# --------------------------------------------------------------------------- #
# 7. Informe de cobertura                                                      #
# --------------------------------------------------------------------------- #
def coverage_report(cargadas: list[TemporadaCargada]) -> pd.DataFrame:
    """Cobertura por temporada, liga y casa. Es la VERIFICACION del esquema.

    Sustituye a la suposicion: en lugar de dar por hecho que Pinnacle o las
    cuotas de cierre existen en toda la muestra, se mide en que subconjunto
    existen realmente. Cualquier analisis debe declarar despues sobre que
    submuestra se ejecuta.
    """
    filas = []
    for c in cargadas:
        base = {"div": c.div, "liga": LIGAS.get(c.div, c.div),
                "temporada": c.temporada, "n_partidos": len(c.partidos)}
        for prefijo in sorted(c.esquema.casas):
            sub = c.cuotas[c.cuotas["prefijo"] == prefijo]
            if sub.empty:
                continue
            fases = set(sub["fase"])
            filas.append({
                **base,
                "casa": CASAS.get(prefijo, (prefijo, False))[0],
                "prefijo": prefijo,
                "es_sharp": CASAS.get(prefijo, (prefijo, False))[1],
                "es_agregado": prefijo in AGREGADOS,
                "mercados": ",".join(sorted(set(sub["mercado"]))),
                "apertura": "apertura" in fases,
                "cierre": "cierre" in fases,
                "n_cuotas": len(sub),
                "cobertura_partidos": round(sub["match_id"].nunique() / max(len(c.partidos), 1), 4),
            })
    return pd.DataFrame(filas)


def resumen_muestra(cargadas: list[TemporadaCargada]) -> dict[str, Any]:
    """Cifras agregadas para declarar el tamano muestral efectivo en el paper."""
    if not cargadas:
        return {"n_partidos": 0}
    partidos = pd.concat([c.partidos for c in cargadas], ignore_index=True)
    cuotas = pd.concat([c.cuotas for c in cargadas], ignore_index=True)
    con_cierre = cuotas[cuotas["fase"] == "cierre"]["match_id"].nunique()
    sharp = cuotas[cuotas["es_sharp"]]["match_id"].nunique()
    return {
        "n_partidos": len(partidos),
        "n_ligas": partidos["div"].nunique(),
        "n_temporadas": partidos["temporada"].nunique(),
        "n_cotizaciones": len(cuotas),
        "n_casas": cuotas[~cuotas["es_agregado"]]["casa"].nunique(),
        "partidos_con_cierre": con_cierre,
        "pct_con_cierre": round(con_cierre / max(len(partidos), 1), 4),
        "partidos_con_sharp": sharp,
        "pct_con_sharp": round(sharp / max(len(partidos), 1), 4),
        "partidos_con_corners": int(partidos["corners_totales"].notna().sum()),
        "partidos_con_tarjetas": int(partidos["tarjetas_totales"].notna().sum()),
        "rango_fechas": [str(partidos["fecha"].min()), str(partidos["fecha"].max())],
    }
