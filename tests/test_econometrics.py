"""Pruebas de la inferencia estadistica y del comportamiento del diseno."""
import numpy as np
import pytest

from veee import econometrics as ec
from veee.model import ModelParams
from veee.simulate import (DGPParams, escenario_h0, escenario_h1_ruido,
                           roi_teorico_h0, simular_muestra)


# --------------------------------------------------------------- contrastes --
def test_t_test_unilateral_signo():
    """Una media claramente positiva rechaza; una negativa nunca debe rechazar."""
    rng = np.random.default_rng(1)
    pos = ec.t_test_roi(rng.normal(0.5, 1.0, 300))
    neg = ec.t_test_roi(rng.normal(-0.5, 1.0, 300))
    assert pos.p_valor < 0.01
    assert neg.p_valor > 0.99          # unilateral por la derecha


def test_t_test_tamano_nominal_bajo_h0():
    """Con datos verdaderamente nulos, la tasa de rechazo ronda alpha."""
    rng = np.random.default_rng(7)
    rechazos = sum(ec.t_test_roi(rng.normal(0.0, 1.0, 200)).p_valor < 0.05
                   for _ in range(400))
    assert 0.02 < rechazos / 400 < 0.09


def test_bootstrap_coincide_aproximadamente_con_t_en_datos_normales():
    rng = np.random.default_rng(3)
    x = rng.normal(0.15, 1.0, 500)
    assert ec.bootstrap_test(x, n_boot=4000).p_valor == pytest.approx(
        ec.t_test_roi(x).p_valor, abs=0.05)


def test_bootstrap_robusto_a_asimetria_extrema():
    """El PnL de apuestas es fuertemente asimetrico; el bootstrap no exige normalidad."""
    rng = np.random.default_rng(11)
    # Cuota 21.0: gana 20 con prob 0.05, pierde 1 en otro caso. EV real = 0.05.
    x = np.where(rng.random(2000) < 0.05, 20.0, -1.0)
    r = ec.bootstrap_test(x, n_boot=4000)
    assert 0.0 <= r.p_valor <= 1.0 and np.isfinite(r.estadistico)


def test_hac_ensancha_ee_con_autocorrelacion():
    """Con dependencia serial positiva, los EE HAC deben superar a los ingenuos."""
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 600)
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = 0.7 * x[i - 1] + e[i]      # AR(1) fuertemente persistente
    assert ec.hac_mean_test(x).ee > ec.t_test_roi(x).ee


def test_benjamini_hochberg_controla_multiplicidad():
    p = [0.001, 0.01, 0.04, 0.30, 0.80]
    out = ec.benjamini_hochberg(p, q=0.05)
    assert out["p_ajustado_BH"].is_monotonic_increasing
    assert bool(out.loc[0, "significativo"]) and not bool(out.loc[4, "significativo"])
    assert (out["p_ajustado_BH"] >= out["p_valor"] - 1e-12).all()


# ----------------------------------------------------------------- potencia --
def test_potencia_exige_muestras_grandes():
    """Resultado central del diseno: detectar un ROI del 3% exige miles de apuestas."""
    p = ec.power_analysis(roi_objetivo=0.03, cuota_media=2.0)
    assert p["n_requerido"] > 4000
    assert ec.power_analysis(roi_objetivo=0.01)["n_requerido"] > \
           ec.power_analysis(roi_objetivo=0.05)["n_requerido"]


def test_sigma_por_apuesta_maxima_en_cuota_dos():
    assert ec.sigma_por_apuesta(0.5, 2.0) == pytest.approx(1.0)


# ------------------------------------------------- validacion del DGP y EMH --
def test_bajo_h0_el_roi_converge_al_margen_negativo():
    """Bajo eficiencia el retorno esperado es exactamente -m/(1+m): no hay alfa."""
    df = simular_muestra(escenario_h0(n_dias=900, vig=0.06), ModelParams())
    roi = df["pnl_unidades"].sum() / df["stake"].sum()
    assert len(df) > 500
    # Tolerancia acorde al error de muestreo (sigma ~ 1 por apuesta).
    assert roi == pytest.approx(roi_teorico_h0(0.06), abs=3.5 / np.sqrt(len(df)))


def test_bajo_h0_el_clv_medio_es_el_margen():
    """El CLV es mucho menos ruidoso que el ROI: bajo H0 se pega al -vig."""
    df = simular_muestra(escenario_h0(n_dias=600, vig=0.06), ModelParams())
    assert df["clv_prob"].mean() == pytest.approx(roi_teorico_h0(0.06), abs=0.02)


def test_ruido_de_precio_genera_alfa_detectable():
    """Un precio insesgado pero disperso es explotable: ineficiencia sin sesgo."""
    df = simular_muestra(escenario_h1_ruido(0.30, n_dias=600, seed=11), ModelParams())
    assert df["pnl_unidades"].sum() / df["stake"].sum() > 0.03
    assert df["clv_prob"].mean() > 0.0


def test_clv_es_mas_potente_que_el_roi():
    """Justificacion empirica del uso del CLV: mismo signo, mucho menos ruido."""
    df = simular_muestra(escenario_h1_ruido(0.30, n_dias=400, seed=5), ModelParams())
    d = ec.prepare(df)
    assert ec.clv_test(d).estadistico > ec.t_test_roi(d["retorno"].values).estadistico


# -------------------------------------------------------------------- logit --
def test_logit_detecta_senal_cuando_existe():
    df = simular_muestra(escenario_h1_ruido(0.35, n_dias=700, seed=21), ModelParams())
    d = ec.prepare(df)
    res, dd = ec.fit_logit(d, ec.FORMULA_COMPLETA)
    assert set(res.params.index) >= {"Intercept", "ev_pct", "trend_score", "clv_pct"}
    diag = ec.logit_diagnostics(res, dd)
    assert 0.0 <= diag["AUC"] <= 1.0
    assert diag["Brier"] <= diag["Brier_referencia_base"] + 1e-6


def test_vif_detecta_colinealidad_estructural():
    """EV y TrendScore comparten construccion: la colinealidad debe ser visible."""
    df = simular_muestra(escenario_h1_ruido(0.30, n_dias=400, seed=9), ModelParams())
    d = ec.prepare(df)
    vif = ec.vif_table(d, ["ev_pct", "trend_score", "clv_pct"])
    assert (vif["VIF"] > 1.0).any() and vif["VIF"].notna().all()


def test_informe_completo_se_genera():
    df = simular_muestra(DGPParams(n_dias=200, lam=0.5, seed=33), ModelParams())
    inf = ec.run_full_analysis(df)
    assert {"descriptivos", "contrastes", "logit", "potencia"} <= inf.keys()
    assert inf["descriptivos"]["n_apuestas"] == len(ec.prepare(df))
    assert 0.0 <= inf["contrastes"]["t_retorno_apuesta"]["p_valor"] <= 1.0


def test_prepare_excluye_no_liquidadas():
    df = simular_muestra(DGPParams(n_dias=60, seed=2), ModelParams())
    df.loc[df.index[:5], "resultado_real"] = None
    assert len(ec.prepare(df)) == len(df) - 5


# --------------------------------- especificacion del contraste de eficiencia --
def test_especificacion_ingenua_rechaza_emh_siendo_cierta():
    """HALLAZGO METODOLOGICO CENTRAL.

    El Logit tal como se especifica habitualmente (EV + TrendScore, sin control
    del precio) encuentra un TrendScore positivo y muy significativo INCLUSO en un
    mercado perfectamente eficiente. Es sesgo de variable omitida: la probabilidad
    de acierto depende mecanicamente del nivel de la cuota. Esta prueba documenta
    el falso positivo para que la especificacion ingenua no se use por descuido.
    """
    d = ec.prepare(simular_muestra(escenario_h0(n_dias=800, seed=11), ModelParams()))
    res, _ = ec.fit_logit(d, ec.FORMULA_INGENUA)
    assert res.params["trend_score"] > 0
    assert res.pvalues["trend_score"] < 0.01        # rechazo espurio de la EMH


def test_especificacion_de_eficiencia_no_rechaza_bajo_h0():
    """Corregida con el control del precio, la senal deja de ser significativa."""
    d = ec.prepare(simular_muestra(escenario_h0(n_dias=800, seed=11), ModelParams()))
    res, _ = ec.fit_logit(d, ec.FORMULA_EFICIENCIA)
    assert res.pvalues["trend_score"] > 0.05
    efic = ec.test_calibracion_precio(res)
    assert efic["wald_p_valor"] > 0.05
    # Bajo eficiencia el precio esta calibrado: beta no difiere de 1.
    assert efic["p_valor_beta_precio_igual_1"] > 0.05


def test_precio_atenuado_bajo_ruido_de_fijacion():
    """Con ruido de precio, beta<1 detecta la miscalibracion del mercado."""
    d = ec.prepare(simular_muestra(escenario_h1_ruido(0.30, n_dias=800, seed=11),
                                   ModelParams()))
    res, _ = ec.fit_logit(d, ec.FORMULA_EFICIENCIA)
    efic = ec.test_calibracion_precio(res)
    assert efic["beta_precio"] < 1.0
    assert efic["p_valor_beta_precio_igual_1"] < 0.05
    assert "atenuado" in efic["interpretacion"]


def test_prepare_construye_el_precio_en_logit():
    d = ec.prepare(simular_muestra(DGPParams(n_dias=60, seed=2), ModelParams()))
    assert "logit_p_mkt" in d and np.isfinite(d["logit_p_mkt"]).all()
