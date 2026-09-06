"""
econometrics.py — Inferencia estadistica sobre la rentabilidad y el modelo Logit.

Contenido
---------
1. Construccion de la serie de ROI (diaria y por apuesta).
2. Contraste t unilateral sobre :math:`\\mu_{ROI}` y sus variantes robustas
   (HAC de Newey-West, errores agrupados por partido, bootstrap por bloques).
3. Regresion logistica de exito con errores estandar agrupados, efectos
   marginales, VIF y diagnostico de calibracion.
4. Analisis de potencia y control de comparaciones multiples.

Advertencias econometricas incorporadas al codigo
-------------------------------------------------
* **No normalidad.** El PnL por apuesta es una variable discreta y fuertemente
  asimetrica a la derecha (pierde 1 con prob. alta, gana c-1 con prob. baja). El
  contraste t es solo asintoticamente valido; por ello se acompana siempre de un
  bootstrap que no impone normalidad.
* **Dependencia.** Varias apuestas del mismo partido comparten choques comunes
  (expulsion, lesion, criterio arbitral). Ignorarlo subestima los errores
  estandar y sobre-rechaza :math:`H_0`. Se agrupa por `match_id`.
* **Endogeneidad del CLV.** El CLV se realiza DESPUES de colocar la apuesta: es
  un *bad control* (Angrist & Pischke, 2009, cap. 3). Se estima el logit en dos
  especificaciones (ex-ante y con CLV como mediador) y se interpretan por separado.
* **Data snooping.** Todo contraste sobre subconjuntos (por mercado) se corrige
  por Benjamini-Hochberg.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 0. Preparacion de la muestra                                                 #
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame, excluir_nulas: bool = False) -> pd.DataFrame:
    """Filtra apuestas liquidadas y crea las variables de trabajo."""
    d = df[df["resultado_real"].notna() & (df["resultado_real"] != "anulada")].copy()
    if excluir_nulas:
        d = d[d["resultado_real"] != "nula"]
    d["fecha"] = pd.to_datetime(d["fecha"]).dt.date
    d["retorno"] = d["pnl_unidades"] / d["stake"]        # ROI por apuesta
    d["ev_pct"] = 100.0 * d["ev_calculado"]
    d["clv_pct"] = 100.0 * d["clv_prob"] if "clv_prob" in d else np.nan
    d["log_cuota"] = np.log(d["cuota_betplay"])
    # Precio neutralizado en escala logit: regresor imprescindible (ver
    # FORMULA_EFICIENCIA). Sin el, el contraste rechaza la EMH espuriamente.
    pm = np.clip(d["p_mercado_devig"].astype(float).values, 1e-9, 1 - 1e-9)
    d["logit_p_mkt"] = np.log(pm / (1 - pm))
    return d.sort_values(["fecha", "id_apuesta"]).reset_index(drop=True)


def daily_roi(d: pd.DataFrame) -> pd.DataFrame:
    r"""Serie de ROI diario.

    .. math:: \mathrm{ROI}_t = \frac{\sum_{i \in t} \mathrm{PnL}_i}{\sum_{i \in t} s_i}

    Con stake plano, el ROI diario es la media de retornos del dia. Es la unidad
    de observacion del contraste principal: agregar por dia reduce (sin eliminar)
    la dependencia intra-jornada y aproxima mejor la normalidad por el TCL.
    """
    g = d.groupby("fecha").agg(
        n=("retorno", "size"), stake=("stake", "sum"), pnl=("pnl_unidades", "sum"),
        ev_medio=("ev_calculado", "mean"), clv_medio=("clv_prob", "mean"),
    )
    g["roi"] = g["pnl"] / g["stake"]
    g["bankroll"] = g["pnl"].cumsum()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# 1. Contrastes sobre la rentabilidad media                                    #
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    nombre: str
    media: float
    ee: float
    estadistico: float
    gl: float | None
    p_valor: float
    ic_95: tuple[float, float]
    n: int
    nota: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def t_test_roi(x: Sequence[float], alpha: float = 0.05) -> TestResult:
    r"""Contraste t de una muestra, unilateral por la derecha.

    .. math::
        H_0:\ \mu_{ROI} \le 0 \qquad \text{frente a} \qquad H_1:\ \mu_{ROI} > 0

    .. math::
        t = \frac{\bar{x} - 0}{s/\sqrt{n}} \ \sim\ t_{n-1} \ \text{bajo } H_0

    Rechazar :math:`H_0` equivale a rechazar la forma debil de la EMH aplicada a
    este mercado: existiria una regla de decision basada exclusivamente en
    informacion publica y pasada con retorno esperado positivo NETO del margen.
    No rechazar es el resultado compatible con eficiencia (y es un resultado
    publicable: la ausencia de alfa es informativa).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        raise ValueError("Se requieren al menos 3 observaciones.")
    media, s = a.mean(), a.std(ddof=1)
    ee = s / np.sqrt(n)
    t = media / ee if ee > 0 else np.inf * np.sign(media)
    p = float(stats.t.sf(t, df=n - 1))                    # cola derecha
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return TestResult("t de Student (unilateral, H1: mu>0)", float(media), float(ee),
                      float(t), n - 1, p,
                      (float(media - tcrit * ee), float(media + tcrit * ee)), n,
                      "Valido asintoticamente; contrastar con bootstrap por asimetria.")


def hac_mean_test(x: Sequence[float], lags: int | None = None) -> TestResult:
    r"""Contraste sobre la media con errores HAC de Newey-West.

    Se estima :math:`x_t = \mu + u_t` por MCO y se corrige la matriz de
    covarianzas por autocorrelacion y heterocedasticidad. Pertinente porque el
    ROI diario puede presentar dependencia serial (mismas jornadas, mismos
    equipos, sesgos persistentes del modelo).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))    # regla de Newey-West
    m = sm.OLS(a, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    media, ee = float(m.params[0]), float(m.bse[0])
    t = media / ee if ee > 0 else np.nan
    p = float(stats.norm.sf(t))
    return TestResult(f"Media con HAC Newey-West (L={max(lags,1)})", media, ee, t, None, p,
                      (media - 1.96 * ee, media + 1.96 * ee), n,
                      "Robusto a heterocedasticidad y autocorrelacion.")


def cluster_mean_test(d: pd.DataFrame, var: str = "retorno",
                      cluster: str = "match_id") -> TestResult:
    """Media con errores estandar agrupados (por partido): corrige dependencia intra-evento."""
    y = d[var].astype(float).values
    m = sm.OLS(y, np.ones(len(y))).fit(cov_type="cluster",
                                       cov_kwds={"groups": d[cluster].values})
    media, ee = float(m.params[0]), float(m.bse[0])
    t = media / ee if ee > 0 else np.nan
    g = d[cluster].nunique()
    p = float(stats.t.sf(t, df=max(g - 1, 1)))
    return TestResult(f"Media con EE agrupados por {cluster}", media, ee, t, g - 1, p,
                      (media - 1.96 * ee, media + 1.96 * ee), len(y),
                      f"{g} conglomerados.")


def bootstrap_test(x: Sequence[float], n_boot: int = 20000, block: int = 1,
                   seed: int = 20260101) -> TestResult:
    r"""Bootstrap (por bloques) del contraste :math:`H_0: \mu \le 0`.

    Se remuestrea la serie **centrada** para imponer :math:`H_0` y se calcula el
    p-valor como la proporcion de replicas cuyo estadistico supera al observado.
    Con `block > 1` se usa un bootstrap por bloques moviles, que preserva la
    dependencia serial. No requiere normalidad: es el contraste de referencia
    dada la fuerte asimetria del PnL de apuestas.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    obs = a.mean()
    centrada = a - obs                                    # imposicion de H0
    if block <= 1:
        reps = rng.choice(centrada, size=(n_boot, n), replace=True).mean(axis=1)
    else:
        n_bloques = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(n_boot, n_bloques))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        reps = centrada[idx].mean(axis=1)
    p = float((reps >= obs).mean())
    ee = float(reps.std(ddof=1))
    lo, hi = np.percentile(a.mean() + reps, [2.5, 97.5])
    return TestResult(f"Bootstrap {'por bloques ' if block > 1 else ''}(B={n_boot})",
                      float(obs), ee, float(obs / ee) if ee > 0 else np.nan, None, p,
                      (float(lo), float(hi)), n,
                      "Libre de supuesto de normalidad; H0 impuesta por centrado.")


def clv_test(d: pd.DataFrame) -> TestResult:
    r"""Contraste sobre el CLV medio: :math:`H_0: \mu_{CLV} \le 0`.

    Es el contraste de **mayor potencia** del estudio. El PnL tiene una varianza
    enorme (una cuota de 2.00 implica sigma ~ 1 por apuesta), de modo que detectar
    un ROI del 3% exige miles de observaciones. El CLV, en cambio, se mide sin el
    ruido del resultado aleatorio: si la estrategia captura informacion genuina,
    debe batir la linea de cierre **aunque** la muestra de PnL sea todavia
    inconcluyente.
    """
    x = d["clv_prob"].dropna().values
    r = t_test_roi(x)
    r.nombre = "CLV medio (t unilateral, H1: mu_CLV>0)"
    r.nota = "Contraste de mayor potencia: aisla la senal del ruido del resultado."
    return r


# --------------------------------------------------------------------------- #
# 2. Modelo Logit                                                              #
# --------------------------------------------------------------------------- #
# Especificacion PRINCIPAL: regresion de eficiencia al estilo Fama. Condiciona
# sobre el precio del mercado, de modo que cualquier poder predictivo residual de
# la senal constituye evidencia contra la eficiencia semifuerte.
FORMULA_EFICIENCIA = "y_exito ~ logit_p_mkt + trend_score"
# Especificacion solicitada en el diseno original, CORREGIDA con el control de
# precio. Sin `logit_p_mkt` el contraste esta mal especificado (ver ADVERTENCIA).
FORMULA_BASE = "y_exito ~ ev_pct + trend_score + logit_p_mkt"
FORMULA_COMPLETA = "y_exito ~ ev_pct + trend_score + clv_pct + logit_p_mkt"
# Especificacion literal del diseno, SIN control de precio. Se estima solo para
# documentar el sesgo de variable omitida; NO debe usarse para inferencia.
FORMULA_INGENUA = "y_exito ~ ev_pct + trend_score + clv_pct"
# Especificacion de eficiencia para el archivo historico: condiciona sobre el
# precio de la casa de referencia (Pinnacle) y contrasta si la discrepancia de
# otra casa aporta poder predictivo. Bajo eficiencia semifuerte, no debe.
FORMULA_ARCHIVO = "y_exito ~ logit_p_sharp + desv_pct"

ADVERTENCIA_ESPECIFICACION = (
    "El Logit sin control del precio de mercado sufre SESGO DE VARIABLE OMITIDA. "
    "La probabilidad de acierto depende mecanicamente del nivel de la cuota, y el "
    "TrendScore esta correlacionado con ella por el propio proceso de seleccion "
    "(solo se apuesta cuando la senal discrepa del precio). En simulaciones bajo "
    "un mercado PERFECTAMENTE EFICIENTE, la especificacion sin control arroja un "
    "coeficiente de TrendScore positivo y significativo al 0.1%: rechazaria la EMH "
    "siendo esta cierta. Toda inferencia debe basarse en FORMULA_EFICIENCIA."
)


def fit_logit(d: pd.DataFrame, formula: str = FORMULA_COMPLETA,
              cluster: str | None = "match_id"):
    r"""Estima el Logit de exito de la apuesta.

    .. math::
        \ln\!\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 \mathrm{EV}_i
        + \beta_2 \mathrm{TrendScore}_i + \beta_3 \mathrm{CLV}_i + \epsilon_i

    Lectura economica de los coeficientes:

    * :math:`\beta_1 > 0` significativo -> el EV construido con las tendencias
      **predice** el exito: hay informacion no incorporada en el precio.
    * :math:`\beta_2 > 0` con :math:`\beta_1` no significativo -> la senal bruta
      informa mas que su traduccion a EV: el fallo esta en el mapeo senal->
      probabilidad, no en la ineficiencia.
    * :math:`\beta_3 > 0` -> confirma que batir la linea de cierre anticipa el
      acierto, validando el CLV como metrica intermedia.
    * Todos no significativos -> evidencia a favor de la EMH en su forma debil.
    """
    d = d[d["y_exito"].notna()].copy()
    d["y_exito"] = d["y_exito"].astype(int)
    cols = [c for c in ["ev_pct", "trend_score", "clv_pct", "logit_p_mkt",
                        "log_cuota", "logit_p_sharp", "desv_pct", "drift_pct"]
            if c in formula]
    d = d.dropna(subset=cols + ["y_exito"])
    if d["y_exito"].nunique() < 2:
        raise ValueError("La variable dependiente no presenta variacion (separacion total).")
    modelo = smf.logit(formula, data=d)
    if cluster and cluster in d:
        res = modelo.fit(disp=False, cov_type="cluster",
                         cov_kwds={"groups": d[cluster].values})
    else:
        res = modelo.fit(disp=False)
    return res, d


def test_calibracion_precio(res) -> dict[str, Any]:
    r"""Contraste de calibracion del precio de mercado en la regresion de eficiencia.

    En la especificacion :math:`\mathrm{logit}(p_i) = \beta_0 + \beta_1
    \mathrm{logit}(p^{mercado}_i) + \beta_2 \mathrm{TrendScore}_i`, la EMH en su
    forma semifuerte implica conjuntamente:

    .. math:: H_0:\ \beta_0 = 0,\quad \beta_1 = 1,\quad \beta_2 = 0

    :math:`\beta_1 = 1` significa que el precio esta perfectamente calibrado;
    :math:`\beta_1 < 1` indica **atenuacion** (el precio es demasiado extremo o
    demasiado ruidoso frente al resultado real), y es la firma tipica del ruido de
    fijacion de precios. :math:`\beta_2 \neq 0` indica informacion publica no
    incorporada. Es el contraste de eficiencia propiamente dicho.
    """
    out: dict[str, Any] = {}
    if "logit_p_mkt" in res.params.index:
        t = res.t_test("logit_p_mkt = 1")
        out["beta_precio"] = float(res.params["logit_p_mkt"])
        out["p_valor_beta_precio_igual_1"] = float(np.squeeze(t.pvalue))
        out["interpretacion"] = ("precio calibrado" if np.squeeze(t.pvalue) > 0.05
                                 else ("precio atenuado (ruido/miscalibracion)"
                                       if res.params["logit_p_mkt"] < 1
                                       else "precio sobre-reactivo"))
    restricciones = [f"{v} = 0" for v in ("trend_score", "ev_pct", "clv_pct")
                     if v in res.params.index]
    if restricciones:
        w = res.wald_test(", ".join(restricciones), scalar=True)
        out["wald_senal_conjunta"] = float(np.squeeze(w.statistic))
        out["wald_p_valor"] = float(np.squeeze(w.pvalue))
        out["restricciones"] = restricciones
        out["conclusion"] = ("No se rechaza la EMH: la senal no aporta poder "
                             "predictivo mas alla del precio."
                             if np.squeeze(w.pvalue) > 0.05 else
                             "Se rechaza la EMH: la senal predice el resultado "
                             "condicionando sobre el precio de mercado.")
    return out


def marginal_effects(res) -> pd.DataFrame:
    """Efectos marginales medios (AME): interpretables en puntos de probabilidad."""
    ame = res.get_margeff(at="overall", method="dydx")
    return pd.DataFrame({
        "AME": ame.margeff, "EE": ame.margeff_se,
        "z": ame.margeff / ame.margeff_se,
        "p": 2 * stats.norm.sf(np.abs(ame.margeff / ame.margeff_se)),
    }, index=ame.exog_names if hasattr(ame, "exog_names") else res.params.index[1:])


def vif_table(d: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    r"""Factores de inflacion de varianza.

    Diagnostico **imprescindible** aqui: EV se construye a partir de p_est, que a
    su vez se construye a partir del TrendScore. La colinealidad entre
    :math:`\beta_1` y :math:`\beta_2` es estructural, no accidental. Si VIF > 10
    los coeficientes individuales son inestables y debe reportarse ademas un
    contraste de Wald conjunto sobre :math:`\beta_1 = \beta_2 = 0`.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(d[list(cols)].dropna())
    return pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })


def logit_diagnostics(res, d: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    """Bondad de ajuste y calibracion: pseudo-R2, Brier, AUC, Hosmer-Lemeshow, linktest."""
    y = d["y_exito"].astype(int).values
    p = res.predict(d).values
    brier = float(np.mean((p - y) ** 2))
    # AUC por el estadistico U de Mann-Whitney (equivalencia exacta).
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = float("nan")
    if n1 and n0:
        r = stats.rankdata(p)
        auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    # Hosmer-Lemeshow.
    hl, gl_hl, p_hl = _hosmer_lemeshow(y, p, bins)
    # Linktest de Pregibon: el cuadrado del indice lineal no debe ser significativo.
    xb = res.fittedvalues if hasattr(res, "fittedvalues") else np.log(p / (1 - p))
    dd = pd.DataFrame({"y": y, "xb": np.asarray(xb), "xb2": np.asarray(xb) ** 2})
    try:
        link = smf.logit("y ~ xb + xb2", data=dd).fit(disp=False)
        p_link = float(link.pvalues["xb2"])
    except Exception:                                     # noqa: BLE001
        p_link = float("nan")
    return {
        "pseudo_R2_McFadden": float(res.prsquared),
        "log_verosimilitud": float(res.llf),
        "LR_p_valor": float(res.llr_pvalue),
        "Brier": brier,
        "Brier_referencia_base": float(np.mean((y.mean() - y) ** 2)),
        "AUC": auc,
        "Hosmer_Lemeshow_chi2": hl, "HL_gl": gl_hl, "HL_p_valor": p_hl,
        "linktest_p_xb2": p_link,
        "n": int(len(y)), "tasa_exito": float(y.mean()),
    }


def _hosmer_lemeshow(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Chi-cuadrado de Hosmer-Lemeshow sobre deciles de riesgo."""
    df = pd.DataFrame({"y": y, "p": p})
    try:
        df["g"] = pd.qcut(df["p"], bins, duplicates="drop")
    except ValueError:
        return float("nan"), 0, float("nan")
    g = df.groupby("g", observed=True).agg(obs=("y", "sum"), n=("y", "size"),
                                           esp=("p", "sum"))
    denom = g["esp"] * (1 - g["esp"] / g["n"])
    stat = float((((g["obs"] - g["esp"]) ** 2) / denom.replace(0, np.nan)).sum())
    gl = max(len(g) - 2, 1)
    return stat, gl, float(stats.chi2.sf(stat, gl))


def calibration_table(res, d: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Tabla de calibracion: probabilidad predicha frente a frecuencia observada."""
    p = res.predict(d).values
    t = pd.DataFrame({"p": p, "y": d["y_exito"].astype(int).values})
    t["bin"] = pd.qcut(t["p"], bins, duplicates="drop")
    out = t.groupby("bin", observed=True).agg(n=("y", "size"), p_media=("p", "mean"),
                                              frec_obs=("y", "mean"))
    out["desvio"] = out["frec_obs"] - out["p_media"]
    return out.reset_index()


# --------------------------------------------------------------------------- #
# 3. Potencia, tamano muestral y comparaciones multiples                       #
# --------------------------------------------------------------------------- #
def sigma_por_apuesta(p: float, cuota: float) -> float:
    r"""Desviacion tipica del retorno de una apuesta binaria.

    .. math:: \sigma = c\,\sqrt{p(1-p)}
    """
    return float(cuota * np.sqrt(p * (1 - p)))


def power_analysis(roi_objetivo: float = 0.03, cuota_media: float = 2.0,
                   alpha: float = 0.05, potencia: float = 0.80) -> dict[str, float]:
    r"""Tamano muestral necesario para detectar un ROI dado.

    .. math:: n \ge \left(\frac{(z_{1-\alpha} + z_{1-\beta})\,\sigma}{\mathrm{ROI}}\right)^2

    Resultado tipico y central para la discusion del paper: con cuota media 2.00
    (sigma ~ 1) detectar un ROI del 3% con potencia 0.80 exige del orden de 6.000
    apuestas. Una temporada de LaLiga no las provee. Por eso el diseno debe
    (i) declarar su potencia ex-ante, (ii) apoyarse en el CLV como contraste
    complementario de mayor potencia, y (iii) abstenerse de interpretar un ROI
    positivo no significativo como evidencia de ineficiencia.
    """
    p = 1.0 / cuota_media
    sigma = sigma_por_apuesta(p, cuota_media)
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(potencia)
    n = (z * sigma / roi_objetivo) ** 2
    return {"roi_objetivo": roi_objetivo, "cuota_media": cuota_media,
            "sigma_por_apuesta": sigma, "n_requerido": float(np.ceil(n)),
            "alpha": alpha, "potencia": potencia,
            "mde_con_1000": float(z * sigma / np.sqrt(1000))}


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05) -> pd.DataFrame:
    """Control de la tasa de falsos descubrimientos (FDR) para contrastes multiples."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    orden = np.argsort(p)
    umbrales = (np.arange(1, m + 1) / m) * q
    signif = np.zeros(m, dtype=bool)
    bajo = p[orden] <= umbrales
    if bajo.any():
        k = np.max(np.where(bajo)[0])
        signif[orden[: k + 1]] = True
    p_adj = np.minimum.accumulate((p[orden] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[orden] = np.clip(p_adj, 0, 1)
    return pd.DataFrame({"p_valor": p, "p_ajustado_BH": out, "significativo": signif})


# --------------------------------------------------------------------------- #
# 4. Orquestador                                                               #
# --------------------------------------------------------------------------- #
def run_full_analysis(df: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Ejecuta la bateria completa de contrastes y devuelve un informe estructurado."""
    d = prepare(df)
    diario = daily_roi(d)
    informe: dict[str, Any] = {
        "descriptivos": {
            "n_apuestas": int(len(d)),
            "n_dias": int(diario.shape[0]),
            "n_partidos": int(d["match_id"].nunique()),
            "cuota_media": float(d["cuota_betplay"].mean()),
            "vig_medio": float(d["overround"].mean()),
            "ev_medio_ex_ante": float(d["ev_calculado"].mean()),
            "tasa_acierto": float(d["y_exito"].dropna().mean()),
            "roi_global": float(d["pnl_unidades"].sum() / d["stake"].sum()),
            "pnl_total": float(d["pnl_unidades"].sum()),
            "clv_medio": float(d["clv_prob"].mean()) if d["clv_prob"].notna().any() else None,
        },
        "contrastes": {},
        "logit": {},
        "potencia": power_analysis(cuota_media=float(d["cuota_betplay"].mean())),
    }

    # --- Contrastes sobre la rentabilidad -------------------------------------
    if len(diario) >= 3:
        informe["contrastes"]["t_roi_diario"] = t_test_roi(diario["roi"].values, alpha).to_dict()
        informe["contrastes"]["hac_roi_diario"] = hac_mean_test(diario["roi"].values).to_dict()
        informe["contrastes"]["bootstrap_roi_diario"] = bootstrap_test(
            diario["roi"].values, block=3).to_dict()
    informe["contrastes"]["t_retorno_apuesta"] = t_test_roi(d["retorno"].values, alpha).to_dict()
    informe["contrastes"]["cluster_retorno"] = cluster_mean_test(d).to_dict()
    if d["clv_prob"].notna().sum() >= 3:
        informe["contrastes"]["clv"] = clv_test(d).to_dict()

    # --- Logit ----------------------------------------------------------------
    informe["advertencia_especificacion"] = ADVERTENCIA_ESPECIFICACION
    for etiqueta, formula in [("eficiencia", FORMULA_EFICIENCIA),
                              ("ex_ante", FORMULA_BASE),
                              ("con_clv", FORMULA_COMPLETA),
                              ("ingenua_NO_USAR", FORMULA_INGENUA)]:
        try:
            res, dd = fit_logit(d, formula)
            informe["logit"][etiqueta] = {
                "formula": formula,
                "coeficientes": res.params.to_dict(),
                "ee_agrupados": res.bse.to_dict(),
                "z": res.tvalues.to_dict(),
                "p_valores": res.pvalues.to_dict(),
                "odds_ratios": np.exp(res.params).to_dict(),
                "ic_95": {k: [float(v[0]), float(v[1])]
                          for k, v in res.conf_int().T.to_dict("list").items()},
                "diagnosticos": logit_diagnostics(res, dd),
                "n": int(dd.shape[0]),
            }
            informe["logit"][etiqueta]["eficiencia"] = test_calibracion_precio(res)
            if etiqueta == "con_clv":
                informe["logit"][etiqueta]["VIF"] = vif_table(
                    dd, [c for c in ["ev_pct", "trend_score", "clv_pct", "logit_p_mkt"]
                         if c in dd]
                ).to_dict("records")
        except Exception as exc:                          # noqa: BLE001
            informe["logit"][etiqueta] = {"error": str(exc)}

    # --- Heterogeneidad por mercado (con correccion FDR) ----------------------
    filas, pv = [], []
    for mercado, sub in d.groupby("mercado"):
        if len(sub) < 10:
            continue
        tr = t_test_roi(sub["retorno"].values)
        filas.append({"mercado": mercado, "n": len(sub),
                      "roi": float(sub["pnl_unidades"].sum() / sub["stake"].sum()),
                      "t": tr.estadistico, "p": tr.p_valor})
        pv.append(tr.p_valor)
    if filas:
        bh = benjamini_hochberg(pv)
        for f, (_, r) in zip(filas, bh.iterrows()):
            f["p_BH"] = float(r["p_ajustado_BH"])
            f["significativo_BH"] = bool(r["significativo"])
        informe["por_mercado"] = filas

    return informe
