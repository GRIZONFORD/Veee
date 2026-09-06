r"""
archive_analysis.py — De cuotas historicas a matriz econometrica.

Convierte la salida de `archive.py` en la matriz que consume `econometrics.py`:
una fila por (partido, casa, mercado, fase, seleccion) con probabilidad
neutralizada, resultado observado y PnL. A partir de ahi, las preguntas del
estudio son agrupaciones sobre esa matriz.

Preguntas que habilita
----------------------
A1  Sesgo favorito-longshot        ROI por decil de probabilidad
A2  **Asimetria entre casas**      ¿la discrepancia frente a la referencia sharp
                                   predice el resultado, o es ruido?
A3  Valor de la linea de cierre    CLV de cada casa frente al cierre de Pinnacle
A4  Desplazamiento apertura-cierre ¿anticipa el movimiento del precio?
A5  Evolucion de la eficiencia     interaccion con la temporada

Identificacion
--------------
Aqui desaparece el problema que lastraba el diseno original. La probabilidad de
referencia proviene de **otra casa** (Pinnacle), no del libro que se evalua, de
modo que es un conjunto de informacion genuinamente independiente. Comparar una
casa consigo misma daba un EV identicamente igual a `-m/(1+m)`; comparar dos
casas distintas no.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .archive import TemporadaCargada
from .oddsmath import devig
from .settlement import liquidar_ou, pnl_ah, pnl_binario

log = logging.getLogger(__name__)

CLAVE_LIBRO = ["match_id", "casa", "mercado", "fase", "linea"]
_MAPA_1X2 = {"H": "home", "D": "draw", "A": "away"}


# --------------------------------------------------------------------------- #
# 1. Neutralizacion del margen libro por libro                                 #
# --------------------------------------------------------------------------- #
def anadir_devig(cuotas: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    r"""Anade la probabilidad neutralizada y el margen de cada libro.

    Se neutraliza **dentro de cada (partido, casa, mercado, fase, linea)**, que es
    la unidad economica correcta: el margen pertenece al libro completo, no a la
    seleccion aislada. Mezclar selecciones de casas distintas produciria
    probabilidades sin sentido.

    Los libros incompletos (una sola seleccion) se conservan con `p_devig` nulo en
    lugar de descartarse: su ausencia debe ser visible en el informe de cobertura.
    """
    out = cuotas.copy()
    out["p_implicita"] = 1.0 / out["cuota"]
    out["p_devig"] = np.nan
    out["overround"] = np.nan
    out["n_selecciones"] = 0

    for _, idx in out.groupby(CLAVE_LIBRO, dropna=False, observed=True).groups.items():
        g = out.loc[idx]
        if len(g) < 2 or g["cuota"].isna().any():
            continue
        try:
            p = devig(g["cuota"].tolist(), method=method)
        except ValueError:
            continue
        out.loc[idx, "p_devig"] = p
        out.loc[idx, "overround"] = float((1.0 / g["cuota"]).sum() - 1.0)
        out.loc[idx, "n_selecciones"] = len(g)
    out["devig_method"] = method
    return out


# --------------------------------------------------------------------------- #
# 2. Liquidacion contra el resultado observado                                 #
# --------------------------------------------------------------------------- #
def liquidar(cuotas: pd.DataFrame, partidos: pd.DataFrame) -> pd.DataFrame:
    """Determina acierto y PnL de cada seleccion con el resultado real.

    Los handicaps asiaticos de cuarto (-0.25, -0.75) se liquidan como media
    victoria o media derrota mediante `settlement.pnl_ah`; tratarlos como binarios
    sesgaria a la vez el PnL y la variable dependiente del logit.
    """
    res = partidos.set_index("match_id")[
        ["resultado_ft", "goles_totales", "margen_goles"]]
    d = cuotas.join(res, on="match_id")

    etiquetas: list[str | None] = []
    pnls: list[float] = []
    for r in d.itertuples(index=False):
        et, pnl = _liquidar_fila(r)
        etiquetas.append(et)
        pnls.append(pnl)

    d["resultado_real"] = etiquetas
    d["pnl_unidades"] = pnls
    d["stake"] = 1.0
    d["y_exito"] = np.where(
        d["resultado_real"].isin(["ganada", "perdida"]),
        (d["resultado_real"] == "ganada").astype(float), np.nan)
    return d


def _liquidar_fila(r) -> tuple[str | None, float]:
    mercado, sel, cuota = r.mercado, r.seleccion, r.cuota
    if mercado == "1X2":
        if not isinstance(r.resultado_ft, str) or r.resultado_ft not in _MAPA_1X2:
            return None, np.nan
        et = "ganada" if sel == _MAPA_1X2[r.resultado_ft] else "perdida"
        return et, pnl_binario(et, cuota)
    if mercado == "ou_2.5":
        if pd.isna(r.goles_totales):
            return None, np.nan
        et = liquidar_ou(float(r.goles_totales), 2.5, sel)
        return et, pnl_binario(et, cuota)
    if mercado == "ah":
        # Una cuota asiatica sin linea es inliquidable: se deja fuera.
        if pd.isna(r.margen_goles) or pd.isna(r.linea):
            return None, np.nan
        margen = float(r.margen_goles)
        linea = float(r.linea)
        if sel == "away":
            margen, linea = -margen, -linea
        try:
            return pnl_ah(margen, linea, cuota)
        except ValueError:
            return None, np.nan
    return None, np.nan


# --------------------------------------------------------------------------- #
# 3. Referencia sharp y asimetria entre casas (A2)                             #
# --------------------------------------------------------------------------- #
CLAVE_SELECCION = ["match_id", "mercado", "seleccion", "linea"]


def anadir_referencia_sharp(d: pd.DataFrame, casa_sharp: str = "Pinnacle",
                            fase_ref: str = "cierre") -> pd.DataFrame:
    r"""Adjunta la probabilidad de la referencia sharp a cada cotizacion.

    La linea de **cierre de Pinnacle** es el mejor estimador disponible de la
    probabilidad real (Bassett, 1981; Woodland & Woodland, 1994): margen minimo y
    ausencia de limitacion al apostador ganador. Es la referencia estandar de la
    literatura.

    La resolucion es **por seleccion, no global**. Las cuotas de cierre solo
    existen en el archivo desde temporadas recientes, de modo que una regla global
    "usar el cierre" dejaria sin referencia a TODAS las temporadas antiguas —que
    si tienen Pinnacle en apertura— y las expulsaria del analisis en silencio.
    Aqui cada seleccion toma el cierre si existe y la apertura en caso contrario,
    y `fase_ref_usada` registra cual se aplico para poder estratificar por ella:
    ambas referencias no son intercambiables y mezclarlas sin declararlo
    confundiria la calidad del estimador con la epoca de la muestra.
    """
    sharp = d[d["casa"] == casa_sharp]
    if sharp.empty:
        log.warning("Referencia sharp '%s' ausente de la muestra.", casa_sharp)
        out = d.copy()
        out["p_sharp"] = np.nan
        out["fase_ref_usada"] = None
        out["desv_vs_sharp"] = np.nan
        return out

    def _ref(fase: str) -> pd.DataFrame:
        sub = sharp[(sharp["fase"] == fase) & sharp["p_devig"].notna()]
        if sub.empty:
            return pd.DataFrame(columns=CLAVE_SELECCION + ["p_sharp"])
        return (sub.groupby(CLAVE_SELECCION, dropna=False, observed=True)["p_devig"]
                   .mean().rename("p_sharp").reset_index())

    principal = _ref(fase_ref)
    alterna = "apertura" if fase_ref == "cierre" else "cierre"
    respaldo = _ref(alterna)

    principal["fase_ref_usada"] = fase_ref
    respaldo["fase_ref_usada"] = alterna
    if not principal.empty and not respaldo.empty:
        # El respaldo solo cubre las selecciones que la referencia principal no
        # alcanza; nunca la sustituye donde esta disponible.
        claves = set(map(tuple, principal[CLAVE_SELECCION].astype(str).values))
        mask = ~respaldo[CLAVE_SELECCION].astype(str).apply(tuple, axis=1).isin(claves)
        respaldo = respaldo[mask]
    ref = pd.concat([principal, respaldo], ignore_index=True)

    if (n_resp := int((ref["fase_ref_usada"] == alterna).sum())):
        log.info("Referencia sharp: %d selecciones usan '%s' por ausencia de '%s'.",
                 n_resp, alterna, fase_ref)

    out = d.merge(ref, on=CLAVE_SELECCION, how="left")
    # Discrepancia frente a la referencia: el regresor central de A2.
    out["desv_vs_sharp"] = out["p_devig"] - out["p_sharp"]
    return out


def dispersion_entre_casas(d: pd.DataFrame) -> pd.DataFrame:
    r"""Dispersion de la probabilidad neutralizada entre casas, por seleccion.

    Es la medida operativa de la **asimetria de cuotas**: cuando las casas
    discrepan sobre el mismo resultado, ¿esa discrepancia contiene informacion o
    es ruido de fijacion de precios? Se excluyen los agregados de mercado
    (Max/Avg), que son estadisticos derivados y no cotizaciones independientes:
    incluirlos contaria dos veces la misma informacion.
    """
    reales = d[~d["es_agregado"] & d["p_devig"].notna()]
    g = reales.groupby(CLAVE_SELECCION + ["fase"], dropna=False, observed=True)
    out = g.agg(n_casas=("p_devig", "size"),
                p_media=("p_devig", "mean"),
                p_std=("p_devig", "std"),
                p_min=("p_devig", "min"),
                p_max=("p_devig", "max"),
                cuota_max=("cuota", "max"),
                cuota_media=("cuota", "mean")).reset_index()
    out["rango_p"] = out["p_max"] - out["p_min"]
    # Ventaja de tomar el mejor precio disponible frente al precio medio: es el
    # limite superior de lo que puede rendir una estrategia de "line shopping".
    out["ventaja_mejor_precio"] = out["cuota_max"] / out["cuota_media"] - 1.0
    return out


# --------------------------------------------------------------------------- #
# 4. Desplazamiento apertura -> cierre (A4)                                    #
# --------------------------------------------------------------------------- #
def desplazamiento_linea(d: pd.DataFrame) -> pd.DataFrame:
    r"""Movimiento del precio de cada casa entre apertura y cierre.

    .. math:: \text{drift} = p^{cierre} - p^{apertura}

    Bajo eficiencia, el desplazamiento recoge la llegada de informacion y por
    tanto **debe** predecir el resultado. Lo que contrasta la eficiencia es si,
    condicionando sobre el precio de cierre, la apertura aporta algo mas: no debe.
    """
    piv = (d[d["p_devig"].notna()]
           .pivot_table(index=CLAVE_SELECCION + ["casa"], columns="fase",
                        values=["p_devig", "cuota"], observed=True))
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    if "p_devig_cierre" not in piv or "p_devig_apertura" not in piv:
        log.warning("Sin ambas fases: no se puede calcular el desplazamiento.")
        return pd.DataFrame()
    piv = piv.dropna(subset=["p_devig_apertura", "p_devig_cierre"])
    piv["drift_p"] = piv["p_devig_cierre"] - piv["p_devig_apertura"]
    piv["drift_cuota"] = piv["cuota_cierre"] / piv["cuota_apertura"] - 1.0
    # CLV de haber tomado la apertura y comparado con el cierre neutralizado.
    piv["clv_prob"] = piv["p_devig_cierre"] * piv["cuota_apertura"] - 1.0
    return piv


# --------------------------------------------------------------------------- #
# 5. Ensamblado                                                                #
# --------------------------------------------------------------------------- #
def build_matrix(cargadas: Iterable[TemporadaCargada], method: str = "shin",
                 casa_sharp: str = "Pinnacle") -> pd.DataFrame:
    """Matriz econometrica completa a partir de varias temporadas cargadas."""
    cargadas = list(cargadas)
    if not cargadas:
        return pd.DataFrame()
    partidos = pd.concat([c.partidos for c in cargadas], ignore_index=True)
    cuotas = pd.concat([c.cuotas for c in cargadas], ignore_index=True)

    meta = partidos.set_index("match_id")[["div", "liga", "temporada", "fecha",
                                           "equipo_local", "equipo_visitante"]]
    d = anadir_devig(cuotas, method=method)
    d = liquidar(d, partidos)
    d = anadir_referencia_sharp(d, casa_sharp)
    d = d.join(meta, on="match_id")

    # Variables listas para `econometrics.py`.
    d["retorno"] = d["pnl_unidades"] / d["stake"]
    d["log_cuota"] = np.log(d["cuota"])
    p = np.clip(d["p_devig"].astype(float), 1e-9, 1 - 1e-9)
    d["logit_p_mkt"] = np.log(p / (1 - p))
    ps = np.clip(d["p_sharp"].astype(float), 1e-9, 1 - 1e-9)
    d["logit_p_sharp"] = np.log(ps / (1 - ps))
    d["desv_pct"] = 100.0 * d["desv_vs_sharp"]
    d["fecha"] = pd.to_datetime(d["fecha"]).dt.date
    return d


def resumen_matriz(d: pd.DataFrame) -> dict[str, object]:
    """Cifras para declarar el tamano muestral efectivo de cada contraste."""
    liq = d[d["y_exito"].notna()]
    return {
        "n_filas": len(d),
        "n_partidos": int(d["match_id"].nunique()),
        "n_liquidadas": len(liq),
        "n_con_devig": int(d["p_devig"].notna().sum()),
        "n_con_sharp": int(d["p_sharp"].notna().sum()),
        "casas": sorted(d["casa"].unique()),
        "mercados": sorted(d["mercado"].unique()),
        "fases": sorted(d["fase"].unique()),
        "overround_medio": float(d["overround"].mean(skipna=True)),
        "roi_global": (float(liq["pnl_unidades"].sum() / liq["stake"].sum())
                       if len(liq) else float("nan")),
    }


# --------------------------------------------------------------------------- #
# 6. Preguntas del estudio (A1-A5)                                             #
# --------------------------------------------------------------------------- #
def a1_sesgo_favorito_longshot(d: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    r"""A1 — ROI por decil de probabilidad neutralizada.

    Bajo eficiencia el ROI debe ser constante e igual a :math:`-m/(1+m)` en todos
    los deciles. El sesgo favorito-longshot se manifiesta como un ROI que
    **decrece** al bajar la probabilidad: los *longshots* rinden peor de lo que su
    precio sugiere (Griffith, 1949; Snowberg & Wolfers, 2010).
    """
    x = d[d["y_exito"].notna() & d["p_devig"].notna() & ~d["es_agregado"]].copy()
    if x.empty:
        return pd.DataFrame()
    x["decil"] = pd.qcut(x["p_devig"], n_deciles, labels=False, duplicates="drop")
    g = x.groupby("decil").agg(
        n=("retorno", "size"), p_media=("p_devig", "mean"),
        cuota_media=("cuota", "mean"), tasa_acierto=("y_exito", "mean"),
        pnl=("pnl_unidades", "sum"), stake=("stake", "sum"))
    g["roi"] = g["pnl"] / g["stake"]
    # Calibracion: diferencia entre lo que el precio promete y lo que ocurre.
    g["sesgo_calibracion"] = g["tasa_acierto"] - g["p_media"]
    return g.reset_index()


def a2_asimetria_entre_casas(d: pd.DataFrame, cluster: str = "match_id"):
    r"""A2 — ¿La discrepancia frente a la referencia sharp predice el resultado?

    .. math::
        \ln\!\left(\frac{p_i}{1-p_i}\right) = \beta_0
        + \beta_1 \ln\!\left(\frac{p^{sharp}_i}{1-p^{sharp}_i}\right)
        + \beta_2 \,\text{desv}_i + \epsilon_i

    Bajo eficiencia semifuerte: :math:`\beta_1 = 1` y :math:`\beta_2 = 0`. Que
    :math:`\beta_2 \neq 0` significaria que la discrepancia entre casas contiene
    informacion sobre el resultado, es decir, **asimetria informativa explotable**.

    Condicionar sobre el precio de referencia es imprescindible: sin ese control,
    la regresion sufre sesgo de variable omitida y rechaza la eficiencia siendo
    cierta (ver `econometrics.ADVERTENCIA_ESPECIFICACION`).
    """
    from .econometrics import FORMULA_ARCHIVO, fit_logit
    x = d[d["y_exito"].notna() & d["p_sharp"].notna() & ~d["es_agregado"]].copy()
    # Se excluye la propia casa de referencia: su desviacion es cero por
    # construccion y solo anadiria masa en el origen.
    x = x[x["desv_vs_sharp"].abs() > 1e-12]
    if x.empty or x["y_exito"].nunique() < 2:
        raise ValueError("Muestra insuficiente para A2.")
    return fit_logit(x, FORMULA_ARCHIVO, cluster=cluster)


def a3_clv_por_casa(d: pd.DataFrame) -> pd.DataFrame:
    r"""A3 — ¿Bate alguna casa la linea de cierre de la referencia?

    .. math:: \mathrm{CLV}_i = p^{sharp}_i \cdot c_i - 1

    Un CLV medio positivo y significativo en una casa concreta significa que sus
    precios ofrecen valor frente al mejor estimador disponible de la probabilidad
    real: es la firma de una casa **soft** explotable por *line shopping*.
    """
    from .econometrics import t_test_roi
    x = d[d["p_sharp"].notna() & d["cuota"].notna()].copy()
    x["clv_prob"] = x["p_sharp"] * x["cuota"] - 1.0
    filas = []
    for (casa, fase), g in x.groupby(["casa", "fase"], observed=True):
        if len(g) < 3:
            continue
        try:
            t = t_test_roi(g["clv_prob"].values)
        except ValueError:
            continue
        filas.append({"casa": casa, "fase": fase, "n": len(g),
                      "clv_medio": t.media, "t": t.estadistico, "p_valor": t.p_valor,
                      "overround_medio": float(g["overround"].mean(skipna=True)),
                      "es_agregado": bool(g["es_agregado"].iloc[0])})
    return pd.DataFrame(filas).sort_values("clv_medio", ascending=False)


def a5_evolucion_por_temporada(d: pd.DataFrame) -> pd.DataFrame:
    """A5 — ¿Mejora la eficiencia con el tiempo?

    Un margen decreciente y una dispersion entre casas menor indican mercados
    mas competitivos. Mezclar epocas sin comprobarlo confundiria el promedio: el
    mercado de 2010 no es el de 2025.
    """
    x = d[~d["es_agregado"]]
    g = x.groupby("temporada").agg(
        n_partidos=("match_id", "nunique"), n_cuotas=("cuota", "size"),
        n_casas=("casa", "nunique"), overround_medio=("overround", "mean"),
        roi=("retorno", "mean"))
    disp = (dispersion_entre_casas(x)
            .merge(x[["match_id", "temporada"]].drop_duplicates(), on="match_id")
            .groupby("temporada")["rango_p"].mean().rename("rango_p_medio"))
    return g.join(disp).reset_index()
