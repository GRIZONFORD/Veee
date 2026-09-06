"""Pruebas del capturador continuo: escalera de ventanas, cierre y salud."""
from datetime import datetime, timedelta, timezone

import pytest

from veee import database as db
from veee.capture import (VENTANAS_H, CaptureParams, finalize_closing_lines,
                          health_check, horas_al_inicio, init_capture_schema,
                          run_capture_cycle, ventana_debida, ventanas_perdidas)
from veee.model import ModelParams

KO = datetime(2026, 9, 12, 19, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cfg(tmp_path):
    return {
        "estudio": {"temporada": "2025-2026", "stake": 1.0},
        "storage": {"db_path": str(tmp_path / "c.db"), "log_dir": str(tmp_path / "l")},
        "betplay": {"liga": "LaLiga"}, "linemate": {"liga": "LaLiga"},
        "fixtures": {"betplay": "data/fixtures/betplay_sample.json",
                     "linemate": "data/fixtures/linemate_sample.json"},
    }


# ------------------------------------------------------ escalera de ventanas --
def test_ventana_debida_elige_el_hito_mas_ajustado():
    assert ventana_debida(50, VENTANAS_H, set()) == 72
    assert ventana_debida(5, VENTANAS_H, set()) == 6
    assert ventana_debida(0.04, VENTANAS_H, set()) == 0.05


def test_no_hay_ventana_antes_del_horizonte_ni_tras_el_inicio():
    assert ventana_debida(80, VENTANAS_H, set()) is None
    assert ventana_debida(-0.5, VENTANAS_H, set()) is None


def test_ventanas_ya_capturadas_no_se_repiten():
    hechas = {72.0, 48.0, 24.0}
    assert ventana_debida(30, VENTANAS_H, hechas) is None
    # A T-20h la ventana de 12h todavia no se ha alcanzado: nada pendiente.
    assert ventana_debida(20, VENTANAS_H, hechas) is None
    # A T-11h si, porque el reloj ya rebaso el hito de 12h.
    assert ventana_debida(11, VENTANAS_H, hechas) == 12


def test_hitos_rebasados_se_marcan_perdidos_no_pendientes():
    """Si cron estuvo caido, los hitos antiguos NO deben dispararse mas tarde:
    etiquetar a T-3h un precio como si fuera de T-24h corromperia el analisis."""
    assert ventana_debida(5, VENTANAS_H, set()) == 6
    assert ventanas_perdidas(5, VENTANAS_H, set()) == [12, 24, 48, 72]
    todas = {6.0, 12.0, 24.0, 48.0, 72.0}
    assert ventana_debida(4.9, VENTANAS_H, todas) is None


def test_horas_al_inicio():
    assert horas_al_inicio(KO.isoformat(), KO - timedelta(hours=6)) == pytest.approx(6.0)
    assert horas_al_inicio(KO.isoformat(), KO + timedelta(hours=1)) == pytest.approx(-1.0)


# ------------------------------------------------------------------- salud ---
def test_salud_detecta_extractor_roto(cfg):
    p = cfg["storage"]["db_path"]
    db.init_db(p); init_capture_schema(p)
    r = health_check(p, "BetPlay", 0, CaptureParams())
    assert r["estado"] == "critico"


def test_salud_alerta_por_caida_de_volumen(cfg):
    p = cfg["storage"]["db_path"]
    db.init_db(p); init_capture_schema(p)
    par = CaptureParams()
    for _ in range(8):
        assert health_check(p, "BetPlay", 100, par)["estado"] == "ok"
    r = health_check(p, "BetPlay", 20, par)      # caida al 20% de la mediana
    assert r["estado"] == "alerta" and r["mediana_ref"] == 100


# -------------------------------------------------- ciclo completo y cierre --
def _timeline(cfg, horas):
    params, par = ModelParams(), CaptureParams()
    out = []
    for h in horas:
        out.append(run_capture_cycle(cfg, params, par, dry_run=True,
                                     ahora=KO - timedelta(hours=h)))
    return out


def test_ciclo_registra_apuestas_solo_en_la_ventana_de_colocacion(cfg):
    """Fuera de [24h, 6h] se evalua pero no se apuesta: el protocolo lo exige."""
    r = _timeline(cfg, [50])[0]                  # T-50h: fuera de ventana
    assert r["n_evaluadas"] > 0
    assert r["n_apuestas_registradas"] == 0
    r = _timeline(cfg, [20])[0]                  # T-20h: dentro
    assert r["n_apuestas_registradas"] > 0


def test_hora_de_colocacion_queda_dentro_de_la_ventana(cfg):
    _timeline(cfg, [50, 30, 20, 10])
    d = db.fetch_df(cfg["storage"]["db_path"], "SELECT horas_al_inicio FROM bets")
    assert len(d) > 0
    assert d["horas_al_inicio"].between(6.0, 24.0).all()


def test_linea_de_cierre_es_el_ultimo_precio_pre_partido(cfg):
    _timeline(cfg, [50, 20, 5, 0.9, 0.04, -0.5])
    p = cfg["storage"]["db_path"]
    cierres = db.fetch_df(p, "SELECT ts_captura, horas_al_inicio FROM odds_snapshots "
                             "WHERE es_cierre = 1")
    assert len(cierres) > 0
    # El cierre corresponde al tick de T-0.04h, no a ninguno anterior.
    assert cierres["horas_al_inicio"].max() < 0.06
    assert (cierres["horas_al_inicio"] > 0).all()      # siempre pre-partido


def test_clv_se_calcula_al_finalizar(cfg):
    _timeline(cfg, [50, 20, 5, 0.04, -0.5])
    d = db.fetch_df(cfg["storage"]["db_path"],
                    "SELECT cuota_betplay, cuota_cierre, clv_odds, clv_prob FROM bets")
    assert d["cuota_cierre"].notna().all()
    assert d["clv_prob"].notna().all()
    # Con cuotas que no se mueven, el CLV en probabilidad es el margen negativo.
    assert d["clv_prob"].between(-0.10, 0.0).all()
    assert d["clv_odds"].abs().max() < 1e-12


def test_precios_en_vivo_no_se_persisten(cfg):
    """Un precio posterior al pitido no es comparable y podria confundirse con cierre."""
    r = _timeline(cfg, [-2.0])[0]
    assert r["precios_descartados_en_vivo"] > 0
    d = db.fetch_df(cfg["storage"]["db_path"],
                    "SELECT horas_al_inicio FROM odds_snapshots")
    assert d.empty or (d["horas_al_inicio"] > -0.1).all()


def test_ciclo_es_idempotente(cfg):
    """Cron cada 5 minutos: reejecutar dentro de la misma ventana no duplica."""
    params, par = ModelParams(), CaptureParams()
    t = KO - timedelta(hours=20)
    run_capture_cycle(cfg, params, par, dry_run=True, ahora=t)
    n1 = db.fetch_df(cfg["storage"]["db_path"], "SELECT COUNT(*) c FROM bets").iloc[0, 0]
    r2 = run_capture_cycle(cfg, params, par, dry_run=True,
                           ahora=t + timedelta(minutes=5))
    n2 = db.fetch_df(cfg["storage"]["db_path"], "SELECT COUNT(*) c FROM bets").iloc[0, 0]
    assert n1 == n2
    assert r2["ventanas_cubiertas"] == 0        # la ventana T-24h ya estaba cubierta


def test_finalize_es_idempotente(cfg):
    _timeline(cfg, [20, 0.04, -0.5])
    p = cfg["storage"]["db_path"]
    r = finalize_closing_lines(p, KO + timedelta(hours=1))
    assert r["apuestas_con_clv"] == 0            # ya estaban calculadas
