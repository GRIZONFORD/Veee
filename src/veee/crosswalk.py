r"""
crosswalk.py — Emparejamiento de equipos y partidos entre fuentes.

El problema
-----------
Cada fuente nombra a los clubes de forma distinta:

    football-data.org      Football-Data.co.uk     una casa de apuestas
    "FC Barcelona"         "Barcelona"             "Barcelona FC"
    "Athletic Club"        "Ath Bilbao"            "Athletic de Bilbao"
    "Deportivo Alavés"     "Alaves"                "Alavés"

Un emparejamiento erroneo **no produce ningun error**: produce una discrepancia
de precio espuria, potencialmente enorme, entre dos partidos distintos. Es el
fallo silencioso mas peligroso de todo el diseno, porque se propaga hasta los
resultados econometricos con apariencia de dato valido.

Estrategia en cascada, de mayor a menor confianza
-------------------------------------------------
1. **Exacto** sobre el nombre tal cual.
2. **Normalizado**: sin acentos, en minusculas y sin los sufijos y prefijos
   societarios (FC, CF, RCD, UD, SD, AC...), que son ruido corporativo y no
   identidad del club.
3. **Difuso** por similitud de cadenas (`difflib`, biblioteca estandar: sin
   dependencias nuevas).

Solo los niveles 1 y 2 se aceptan automaticamente. **Todo emparejamiento difuso
queda marcado para revision humana**, y los no emparejados se listan
explicitamente. Automatizar el nivel 3 sin supervision es exactamente como se
cuelan los errores que luego nadie encuentra.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

log = logging.getLogger(__name__)

# Ruido societario: identifica la forma juridica, no al club.
_AFIJOS = {
    "fc", "cf", "sc", "ac", "afc", "cd", "ud", "sd", "rc", "rcd", "cp",
    "club", "cfc", "bc", "ss", "as", "us", "aс", "calcio", "futbol", "fútbol",
    "de", "the", "1899", "1904", "1909", "05", "04", "1846",
}
_RE_NO_ALFA = re.compile(r"[^a-z0-9\s]")

# Casos que ninguna heuristica resuelve: nombres historicos o coloquiales que no
# comparten raiz. Se declaran a mano y se versionan con el codigo.
ALIAS_MANUALES: dict[str, str] = {
    # Clave y valor van en el espacio YA normalizado (sin acentos, sin afijos).
    "ath bilbao": "athletic", "athletic bilbao": "athletic",
    "ath madrid": "atletico madrid", "atletico": "atletico madrid",
    "espanol": "espanyol", "sp gijon": "sporting gijon",
    "la coruna": "deportivo la coruna",
    "man united": "manchester united", "man city": "manchester city",
    "nottm forest": "nottingham forest", "sheffield weds": "sheffield wednesday",
    "wolves": "wolverhampton wanderers", "spurs": "tottenham hotspur",
    "west brom": "west bromwich albion", "west ham": "west ham united",
    "inter": "internazionale", "milan": "milan",
    "psg": "paris saint germain", "leverkusen": "bayer leverkusen",
    "dortmund": "borussia dortmund", "bayern munich": "bayern munchen",
    "m gladbach": "borussia monchengladbach",
}


def normalizar_nombre(nombre: str) -> str:
    """Reduce un nombre de club a su nucleo identificativo.

    Retira acentos, puntuacion y los afijos societarios, y despues aplica los
    alias manuales. El orden importa: los alias se definen sobre la forma ya
    normalizada, porque 'Athletic Club' pierde 'club' como afijo y se convierte
    en 'athletic'.
    """
    if not isinstance(nombre, str):
        return ""
    s = unicodedata.normalize("NFKD", nombre)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _RE_NO_ALFA.sub(" ", s)
    tokens = [t for t in s.split() if t and t not in _AFIJOS]
    base = " ".join(tokens).strip()
    return ALIAS_MANUALES.get(base, base)


def _tokens(nombre_norm: str) -> frozenset[str]:
    return frozenset(nombre_norm.split())


def _similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def emparejar_equipos(origen: list[str], destino: list[str],
                      umbral_difuso: float = 0.82) -> pd.DataFrame:
    """Empareja dos listas de nombres de club mediante una cascada de 4 niveles.

    Devuelve una fila por nombre de `origen` con su correspondencia, el metodo
    empleado, la puntuacion y si **requiere revision humana**.

    Trampa deliberadamente contemplada: 'Barcelona' es subconjunto tanto de
    'FC Barcelona' como de 'RCD Espanyol de Barcelona'. Un emparejamiento por
    subconjunto ingenuo confundiria dos clubes rivales de la misma ciudad y
    contaminaria el estudio con partidos equivocados. Por eso **toda coincidencia
    ambigua se marca para revision en lugar de resolverse por conjetura**.
    """
    dest_norm = {d: normalizar_nombre(d) for d in destino}
    dest_tok = {d: _tokens(n) for d, n in dest_norm.items()}
    filas = []

    for o in origen:
        o_norm = normalizar_nombre(o)
        o_tok = _tokens(o_norm)
        metodo, destino_final, score, nota = "sin_emparejar", None, 0.0, ""

        if o in destino:
            metodo, destino_final, score = "exacto", o, 1.0
        elif (ex := [d for d, dn in dest_norm.items() if dn == o_norm and dn]):
            if len(ex) > 1:
                metodo, score, nota = "ambiguo", 1.0, f"coincide con {ex}"
            else:
                metodo, destino_final, score = "normalizado", ex[0], 1.0
        else:
            # Nivel 3: subconjunto de tokens en cualquier direccion.
            #   'sociedad' subset de 'real sociedad'  -> mismo club
            #   'alaves'   subset de 'deportivo alaves'
            subs = [d for d, dt in dest_tok.items()
                    if dt and o_tok and (dt <= o_tok or o_tok <= dt)]
            if len(subs) == 1:
                metodo, destino_final, score = "subconjunto", subs[0], 0.95
            elif len(subs) > 1:
                metodo, score, nota = "ambiguo", 0.95, f"coincide con {subs}"
            else:
                # Nivel 4: similitud de cadenas, solo si hay un ganador claro.
                puntuadas = sorted(((_similitud(o_norm, dn), d)
                                    for d, dn in dest_norm.items()), reverse=True)
                if puntuadas and puntuadas[0][0] >= umbral_difuso:
                    mejor_s, mejor_d = puntuadas[0]
                    segundo = puntuadas[1][0] if len(puntuadas) > 1 else 0.0
                    if mejor_s - segundo < 0.05:
                        metodo, score = "ambiguo", mejor_s
                        nota = f"empate difuso con {[d for _, d in puntuadas[:2]]}"
                    else:
                        metodo, destino_final, score = "difuso", mejor_d, mejor_s
                elif puntuadas:
                    score = puntuadas[0][0]

        filas.append({
            "origen": o, "origen_norm": o_norm,
            "destino": destino_final, "metodo": metodo, "score": round(score, 4),
            "nota": nota,
            # Exacto, normalizado y subconjunto unico se aceptan sin supervision.
            # Difuso, ambiguo y sin emparejar, nunca.
            "requiere_revision": metodo in ("difuso", "ambiguo", "sin_emparejar"),
        })

    df = pd.DataFrame(filas)
    if (n_rev := int(df["requiere_revision"].sum())):
        log.warning("%d de %d equipos requieren revision humana.", n_rev, len(df))
    return df


def informe_emparejamiento(df: pd.DataFrame) -> dict[str, object]:
    """Resumen para decidir si el cruce es utilizable."""
    total = len(df)
    por_metodo = df["metodo"].value_counts().to_dict()
    automaticos = int(df[~df["requiere_revision"]].shape[0])
    return {
        "n_equipos": total,
        "automaticos": automaticos,
        "cobertura_automatica": round(automaticos / total, 4) if total else 0.0,
        "por_metodo": por_metodo,
        "a_revisar": df[df["requiere_revision"]][
            ["origen", "destino", "metodo", "score"]].to_dict("records"),
    }


def emparejar_partidos(a: pd.DataFrame, b: pd.DataFrame,
                       mapa_equipos: dict[str, str] | None = None,
                       tolerancia_dias: int = 1) -> pd.DataFrame:
    """Empareja partidos de dos fuentes por (fecha, local, visitante).

    La tolerancia de un dia es necesaria y no un capricho: un partido a las 21:00
    en Madrid cae en el dia siguiente en UTC segun la fuente, y una fuente que
    registre en hora local y otra en UTC discreparian sistematicamente en la
    fecha sin que ningun nombre este mal.
    """
    mapa = mapa_equipos or {}

    def clave(df: pd.DataFrame, local: str, visita: str) -> pd.DataFrame:
        out = df.copy()
        out["_local"] = out[local].map(lambda x: mapa.get(x, normalizar_nombre(x)))
        out["_visita"] = out[visita].map(lambda x: mapa.get(x, normalizar_nombre(x)))
        out["_fecha"] = pd.to_datetime(out["fecha"], errors="coerce", utc=True).dt.tz_localize(None)
        return out

    A = clave(a, "equipo_local", "equipo_visitante")
    B = clave(b, "equipo_local", "equipo_visitante")

    fusion = A.merge(B, on=["_local", "_visita"], how="inner",
                     suffixes=("_a", "_b"))
    if fusion.empty:
        log.warning("Cruce vacio: revise el emparejamiento de nombres de equipo.")
        return fusion
    delta = (fusion["_fecha_a"] - fusion["_fecha_b"]).abs().dt.days
    fusion = fusion[delta <= tolerancia_dias].copy()
    fusion["dias_diferencia"] = delta[delta <= tolerancia_dias]
    return fusion
