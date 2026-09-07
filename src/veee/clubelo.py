r"""
clubelo.py — Conector de ClubElo y estimador p_elo del resultado 1X2.

Por que ClubElo y no otra fuente de fuerza de equipo
-----------------------------------------------------
De las fuentes investigadas en docs/PLAN_ALINEACIONES.md, ClubElo es la unica
que combina tres propiedades a la vez: (1) API HTTP simple, sin navegador ni
autenticacion; (2) historica desde 1939 y consultable a fecha exacta, de modo
que se puede enriquecer RETROACTIVAMENTE toda la Via A sin recolectar nada en
vivo; (3) un unico numero por club y fecha, directamente vinculado a la
probabilidad de resultado, sin la ambiguedad de agregar decenas de atributos de
plantilla.

Formato del API (verificado leyendo el codigo fuente de `soccerdata`, no solo
su documentacion, porque `api.clubelo.com` esta bloqueado en este entorno):

    http://api.clubelo.com/{YYYY-MM-DD}   -> foto de TODOS los clubes esa fecha
    http://api.clubelo.com/{NombreClub}   -> historial COMPLETO de un club

Columnas: ``Rank,Club,Country,Level,Elo,From,To``. Cada fila es un INTERVALO de
validez: el Elo de un club cambia tras cada partido, de modo que ``From``/``To``
delimitan el periodo en que ese valor estuvo vigente. Obtener "el Elo a fecha
X" es una busqueda de intervalo (``From <= X <= To``), nunca tomar el valor mas
reciente sin mas.

El problema de identificacion de nombres (otra vez)
----------------------------------------------------
ClubElo usa su propia grafia de club, distinta de Football-Data.co.uk y de
football-data.org. Es EXACTAMENTE el mismo problema que resolvio
`crosswalk.py` para football-data.org: un emparejamiento erroneo no falla, deja
un Elo silenciosamente equivocado unido a un partido. Este modulo reutiliza
`crosswalk.emparejar_equipos` en vez de inventar una segunda heuristica.

Disciplina anti-fuga (leakage)
-------------------------------
`elo_as_of` exige una fecha explicita y busca el intervalo vigente en ESA
fecha. Tomar "el Elo de hoy" para enriquecer un partido de 2015 seria fuga de
datos identica en espiritu a la que se documento en docs/PLAN_ALINEACIONES.md
sobre el `read_lineup()` de FBref: usar informacion que no existia en el
momento que se pretende modelar.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLUB_ELO_API = "http://api.clubelo.com"
COLUMNAS_ESPERADAS = ["Rank", "Club", "Country", "Level", "Elo", "From", "To"]


# --------------------------------------------------------------------------- #
# 1. Lectura robusta del CSV                                                   #
# --------------------------------------------------------------------------- #
def parse_csv_robusto(datos: bytes) -> pd.DataFrame:
    """Valida el formato en vez de suponerlo.

    Si `api.clubelo.com` cambiara su esquema, un `pd.read_csv` ingenuo seguido
    de acceso directo a columnas fallaria mas adelante con un KeyError opaco, o
    peor, produciria NaN silenciosos. Aqui se comprueba explicitamente que las
    columnas documentadas existan antes de seguir.
    """
    df = pd.read_csv(io.BytesIO(datos))
    faltan = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if faltan:
        raise ValueError(
            f"Formato de ClubElo inesperado: faltan columnas {sorted(faltan)}. "
            f"Columnas recibidas: {list(df.columns)}. "
            "El API pudo haber cambiado; revise COLUMNAS_ESPERADAS."
        )
    out = df.rename(columns=str.lower)
    out["from"] = pd.to_datetime(out["from"], errors="coerce")
    out["to"] = pd.to_datetime(out["to"], errors="coerce")
    out["elo"] = pd.to_numeric(out["elo"], errors="coerce")
    malas_fechas = out["from"].isna() | out["to"].isna()
    if malas_fechas.any():
        log.warning("%d filas con From/To no interpretable; se descartan.",
                    int(malas_fechas.sum()))
        out = out[~malas_fechas]
    invertido = out["from"] > out["to"]
    if invertido.any():
        log.warning("%d filas con From > To (dato corrupto); se descartan.",
                    int(invertido.sum()))
        out = out[~invertido]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Descarga y cache                                                          #
# --------------------------------------------------------------------------- #
def _slug(nombre: str) -> str:
    """Nombre de fichero seguro para un club (ClubElo usa espacios y guiones)."""
    return "".join(c if c.isalnum() else "_" for c in nombre).strip("_")


def descargar_historial(equipo_clubelo: str, cache_dir: str | Path = "data/clubelo",
                        session: Any = None, timeout: int = 10,
                        max_retries: int = 2, forzar: bool = False) -> Path:
    """Descarga el historial COMPLETO de un club (una fila por intervalo de Elo).

    `equipo_clubelo` debe ser la grafia exacta que usa ClubElo (ver
    `crosswalk.py` / `mapa_desde_snapshot`), no el nombre del archivo CSV ni el
    de football-data.org.
    """
    destino = Path(cache_dir) / "historial" / f"{_slug(equipo_clubelo)}.csv"
    if destino.exists() and not forzar:
        log.debug("Cache: %s", destino)
        return destino
    from .scrapers.base import PoliteSession
    session = session or PoliteSession(rate_limit_s=1.0, cache_dir=cache_dir,
                                       timeout=timeout, max_retries=max_retries)
    r = session.get(f"{CLUB_ELO_API}/{equipo_clubelo}", cache=False)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    log.info("Descargado historial de %s (%d bytes)", equipo_clubelo, len(r.content))
    return destino


def descargar_snapshot(fecha: str | datetime, cache_dir: str | Path = "data/clubelo",
                       session: Any = None, timeout: int = 10,
                       max_retries: int = 2, forzar: bool = False) -> Path:
    """Descarga la foto de TODOS los clubes en una fecha (para construir el
    crosswalk de nombres: revela la grafia de ClubElo de cada club vigente)."""
    dia = fecha if isinstance(fecha, str) else fecha.strftime("%Y-%m-%d")
    destino = Path(cache_dir) / "snapshots" / f"{dia}.csv"
    if destino.exists() and not forzar:
        return destino
    from .scrapers.base import PoliteSession
    session = session or PoliteSession(rate_limit_s=1.0, cache_dir=cache_dir,
                                       timeout=timeout, max_retries=max_retries)
    r = session.get(f"{CLUB_ELO_API}/{dia}", cache=False)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    log.info("Descargado snapshot %s (%d bytes)", dia, len(r.content))
    return destino


def probar_conexion(cache_dir: str | Path = "data/clubelo", timeout: int = 8,
                    max_retries: int = 1) -> tuple[bool, str]:
    """Prueba de humo rapida: un club conocido, para fallar en segundos y no en
    minutos si el host esta inalcanzable (mismo patron que `archive.probar_conexion`)."""
    from .scrapers.base import PoliteSession
    sesion = PoliteSession(rate_limit_s=0, cache_dir=cache_dir,
                           timeout=timeout, max_retries=max_retries)
    try:
        r = sesion.get(f"{CLUB_ELO_API}/Barcelona", cache=False)
        parse_csv_robusto(r.content)          # valida tambien el esquema
        return True, f"OK ({len(r.content):,} bytes, esquema valido)"
    except Exception as exc:                  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def cargar_historial(ruta: str | Path) -> pd.DataFrame:
    return parse_csv_robusto(Path(ruta).read_bytes())


def cargar_snapshot(ruta: str | Path) -> pd.DataFrame:
    return parse_csv_robusto(Path(ruta).read_bytes())


# --------------------------------------------------------------------------- #
# 3. Busqueda "a fecha" (as-of) e integracion con el archivo                   #
# --------------------------------------------------------------------------- #
def elo_as_of(historial: pd.DataFrame, fecha: pd.Timestamp) -> tuple[float, str]:
    r"""Elo vigente de un club en una fecha exacta.

    Busca el intervalo `[From, To]` que contiene `fecha`. Devuelve
    `(nan, "sin_cobertura")` si la fecha cae antes del primer registro, despues
    del ultimo, o en un hueco entre intervalos — nunca extrapola ni toma el
    valor mas cercano en silencio: **una fecha sin Elo es NaN, no una
    aproximacion**.
    """
    if historial.empty or pd.isna(fecha):
        return float("nan"), "sin_datos"
    fila = historial[(historial["from"] <= fecha) & (fecha <= historial["to"])]
    if fila.empty:
        antes = historial["to"] < fecha
        motivo = ("anterior_al_primer_registro" if not antes.any()
                  else "posterior_al_ultimo_registro" if antes.all()
                  else "hueco_entre_intervalos")
        return float("nan"), motivo
    if len(fila) > 1:
        # Intervalos solapados no deberian ocurrir; se toma el mas reciente y
        # se deja constancia en vez de fallar silenciosamente.
        log.warning("Intervalos de Elo solapados en fecha %s; se usa el ultimo.", fecha)
        fila = fila.sort_values("from").tail(1)
    return float(fila["elo"].iloc[0]), "ok"


def enriquecer_con_elo(partidos: pd.DataFrame, historiales: dict[str, pd.DataFrame],
                       mapa_local: dict[str, str], mapa_visitante: dict[str, str] | None = None,
                       col_fecha: str = "fecha") -> pd.DataFrame:
    r"""Adjunta `elo_local`, `elo_visitante` y `elo_diff` a una tabla de partidos.

    `mapa_local`/`mapa_visitante` traducen el nombre de equipo del partido (el
    del archivo historico) a la clave usada en `historiales` (el nombre de
    ClubElo). Si no se pasa `mapa_visitante`, se reutiliza `mapa_local` para
    ambas columnas: en la practica es el mismo diccionario de equipo->ClubElo
    aplicado a `equipo_local` y a `equipo_visitante`.

    `elo_diff = elo_local - elo_visitante` es el UNICO regresor de
    `ajustar_modelo_elo`: con un logit ordenado de una sola variable, la
    ventaja de jugar en casa queda absorbida por los puntos de corte estimados,
    sin necesidad de anadir una constante de ventaja de local a mano.
    """
    mapa_visitante = mapa_visitante or mapa_local
    out = partidos.copy()
    fechas = pd.to_datetime(out[col_fecha], errors="coerce")

    def _buscar(equipos: pd.Series, mapa: dict[str, str]) -> tuple[list[float], list[str]]:
        elos, motivos = [], []
        for nombre, fecha in zip(equipos, fechas):
            clave = mapa.get(nombre)
            hist = historiales.get(clave) if clave else None
            if hist is None:
                elos.append(float("nan")); motivos.append("sin_mapa_clubelo")
                continue
            e, m = elo_as_of(hist, fecha)
            elos.append(e); motivos.append(m)
        return elos, motivos

    out["elo_local"], out["elo_local_estado"] = _buscar(out["equipo_local"], mapa_local)
    out["elo_visitante"], out["elo_visitante_estado"] = _buscar(out["equipo_visitante"], mapa_visitante)
    out["elo_diff"] = out["elo_local"] - out["elo_visitante"]
    out["elo_cobertura"] = (out["elo_local_estado"] == "ok") & (out["elo_visitante_estado"] == "ok")
    return out


# --------------------------------------------------------------------------- #
# 4. Estimador p_elo: logit ordenado del resultado 1X2                        #
# --------------------------------------------------------------------------- #
ORDEN_RESULTADO = {"A": 0, "D": 1, "H": 2}   # away < draw < home, categoria ordenada


@dataclass
class ModeloElo:
    """Logit ordenado calibrado: resultado ~ elo_diff.

    Se guarda el propio objeto ajustado de statsmodels (`ajuste`) junto con los
    metadatos de calibracion, para que quede trazable con que muestra y en que
    fecha se ajusto — igual que `model_version` en el resto del repo.
    """
    ajuste: Any
    n: int
    rango_fechas: tuple[str, str]
    ajustado_en: str = field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))

    def predecir(self, elo_diff: float | np.ndarray) -> pd.DataFrame:
        r"""P(away), P(draw), P(home) para una o varias diferencias de Elo."""
        x = np.atleast_1d(np.asarray(elo_diff, dtype=float))
        probs = self.ajuste.model.predict(self.ajuste.params, exog=x.reshape(-1, 1))
        return pd.DataFrame(probs, columns=["p_away", "p_draw", "p_home"])


def ajustar_modelo_elo(partidos_enriquecidos: pd.DataFrame,
                       col_resultado: str = "resultado_ft") -> ModeloElo:
    r"""Ajusta un logit ordenado: resultado (away<draw<home) ~ elo_diff.

    .. math::
        P(Y \le j \mid \Delta\text{Elo}) = \frac{1}{1+\exp(-(\kappa_j - \beta\,\Delta\text{Elo}))}

    con :math:`\Delta\text{Elo} = \text{Elo}_{local} - \text{Elo}_{visitante}` y
    :math:`\kappa_j` los puntos de corte estimados entre las tres categorias.
    Es el enfoque estandar en la literatura (Hvattum & Arntzen, 2010) para
    convertir una diferencia de Elo en una distribucion de resultado 1X2 en vez
    de un simple binario gana/no-gana: una sola variable basta porque la
    ventaja de jugar en casa se manifiesta como una asimetria en los puntos de
    corte, no como un termino aparte.

    ADVERTENCIA DE DISCIPLINA: este ajuste debe hacerse SOLO sobre partidos
    anteriores a los que se pretende evaluar (particion temporal, igual que en
    `econometrics.py`). Ajustar con toda la muestra y evaluar sobre la misma
    muestra es la fuga de datos mas basica y mas facil de cometer por descuido.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = partidos_enriquecidos[
        partidos_enriquecidos["elo_cobertura"]
        & partidos_enriquecidos[col_resultado].isin(ORDEN_RESULTADO)
    ].copy()
    if len(d) < 10:
        raise ValueError(
            f"Solo {len(d)} partidos con Elo y resultado validos: insuficiente "
            "para calibrar. Amplie la cobertura de ClubElo o el rango de fechas."
        )
    y = d[col_resultado].map(ORDEN_RESULTADO).astype(int)
    x = d[["elo_diff"]].astype(float)
    modelo = OrderedModel(y, x, distr="logit")
    ajuste = modelo.fit(method="bfgs", disp=False)
    fechas = pd.to_datetime(d["fecha"], errors="coerce").dropna()
    rango = (str(fechas.min().date()), str(fechas.max().date())) if len(fechas) else ("", "")
    return ModeloElo(ajuste, len(d), rango)


def evaluar_modelo_elo(modelo: ModeloElo, partidos_enriquecidos: pd.DataFrame,
                       col_resultado: str = "resultado_ft") -> dict[str, Any]:
    """Brier score (multiclase) y comparacion contra el precio de mercado si esta
    disponible, reutilizando exactamente las metricas ya usadas en el resto del
    estudio en vez de inventar unas nuevas."""
    d = partidos_enriquecidos[
        partidos_enriquecidos["elo_cobertura"]
        & partidos_enriquecidos[col_resultado].isin(ORDEN_RESULTADO)
    ].copy()
    pred = modelo.predecir(d["elo_diff"].values)
    y_idx = d[col_resultado].map(ORDEN_RESULTADO).astype(int).values
    y_onehot = np.eye(3)[y_idx]
    brier_elo = float(np.mean(np.sum((pred.values - y_onehot) ** 2, axis=1)))
    # Brier de referencia: predecir siempre la frecuencia marginal observada.
    marginal = y_onehot.mean(axis=0)
    brier_base = float(np.mean(np.sum((marginal - y_onehot) ** 2, axis=1)))

    salida: dict[str, Any] = {
        "n": len(d), "brier_elo": brier_elo, "brier_base_marginal": brier_base,
        "mejora_vs_marginal": brier_base - brier_elo,
        "log_verosimilitud": float(modelo.ajuste.llf),
        "pseudo_R2_McFadden": float(modelo.ajuste.prsquared)
                              if hasattr(modelo.ajuste, "prsquared") else None,
        # `statsmodels.OrderedModel` guarda los puntos de corte en una
        # parametrizacion de optimizacion (el primero directo, los siguientes
        # como incrementos exponenciados) que NO es comparable sin mas a un
        # corte derivado a mano. `transform_threshold_params` los devuelve ya
        # en la escala real del indice lineal.
        # Se descartan los extremos -inf/+inf (bordes convencionales de la
        # primera y ultima categoria): solo interesan los dos cortes internos
        # away|draw y draw|home.
        "puntos_de_corte": [round(float(k), 4) for k in
                            modelo.ajuste.model.transform_threshold_params(
                                modelo.ajuste.params.values)[1:-1]],
        "beta_elo_diff": float(modelo.ajuste.params.get("elo_diff", float("nan")))
                         if hasattr(modelo.ajuste.params, "get") else None,
    }
    return salida


def comparar_con_mercado(pred_elo: pd.DataFrame, resultado_ft: pd.Series,
                         p_mercado: pd.Series, seleccion: pd.Series) -> dict[str, float]:
    r"""Brier de p_elo frente al Brier del precio de mercado, sobre la MISMA
    muestra y seleccion. Es la pregunta que de verdad importa: no si el modelo
    de Elo "funciona" en abstracto, sino si aporta algo que el precio no tenga
    ya incorporado — la misma logica de identificacion que el resto del estudio.

    `pred_elo` debe tener el mismo indice posicional (0..n-1) que `resultado_ft`,
    `p_mercado` y `seleccion` (reset_index antes de llamar si hace falta).
    """
    mapa_col = {"home": "p_home", "draw": "p_draw", "away": "p_away"}
    y = (seleccion.map({"home": "H", "draw": "D", "away": "A"}) == resultado_ft.values
         ).astype(float).values
    brier_mkt = float(np.mean((p_mercado.values.astype(float) - y) ** 2))
    # DataFrame.lookup fue retirado en pandas 2.0; se indexa por posicion.
    cols = seleccion.map(mapa_col).values
    col_idx = np.array([pred_elo.columns.get_loc(c) for c in cols])
    p_elo_sel = pred_elo.values[np.arange(len(pred_elo)), col_idx]
    brier_elo = float(np.mean((p_elo_sel - y) ** 2))
    return {"n": len(y), "brier_mercado": brier_mkt, "brier_elo": brier_elo,
           "elo_mejor_que_mercado": brier_elo < brier_mkt}


# --------------------------------------------------------------------------- #
# 5. Crosswalk de nombres (reutiliza crosswalk.py)                            #
# --------------------------------------------------------------------------- #
def mapa_desde_snapshot(snapshot: pd.DataFrame, nombres_archivo: Iterable[str],
                        pais: str | None = "ESP") -> pd.DataFrame:
    """Empareja los nombres del archivo historico con la grafia de ClubElo.

    Reutiliza `crosswalk.emparejar_equipos`: exacto/normalizado/subconjunto se
    aceptan solos; lo difuso o ambiguo se marca para revision humana, con el
    mismo criterio que ya protege el cruce con football-data.org.
    """
    from .crosswalk import emparejar_equipos
    candidatos = snapshot
    if pais and "country" in snapshot.columns:
        acotado = snapshot[snapshot["country"] == pais]
        if not acotado.empty:
            candidatos = acotado
    return emparejar_equipos(list(nombres_archivo), sorted(candidatos["club"].unique()))
