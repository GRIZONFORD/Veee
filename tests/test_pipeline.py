"""Prueba de integracion: ciclo completo extraccion -> registro -> liquidacion."""
import json

import pytest

from veee import database as db
from veee.model import ModelParams
from veee.pipeline import run_daily
from veee.settlement import capturar_cierre, liquidar_partido


@pytest.fixture()
def cfg(tmp_path):
    return {
        "estudio": {"temporada": "2025-2026", "stake": 1.0},
        "storage": {"db_path": str(tmp_path / "t.db"),
                    "log_dir": str(tmp_path / "logs")},
        "betplay": {"liga": "LaLiga"}, "linemate": {"liga": "LaLiga"},
        "fixtures": {"betplay": "data/fixtures/betplay_sample.json",
                     "linemate": "data/fixtures/linemate_sample.json"},
    }


def test_ciclo_diario_registra_apuestas(cfg):
    r = run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    assert r["n_cuotas"] > 0 and r["n_tendencias"] > 0
    assert r["n_apuestas_registradas"] >= 1
    d = db.fetch_df(cfg["storage"]["db_path"], "SELECT * FROM bets")
    assert (d["ev_calculado"] > 0).all()          # solo se registran +EV
    assert d["p_estimada"].between(0, 1).all()


def test_ciclo_es_idempotente(cfg):
    """Reejecutar el mismo dia no debe duplicar ni alterar decisiones."""
    r1 = run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    n1 = db.fetch_df(cfg["storage"]["db_path"], "SELECT * FROM bets").shape[0]
    r2 = run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    n2 = db.fetch_df(cfg["storage"]["db_path"], "SELECT * FROM bets").shape[0]
    assert n1 == n2 and r2["n_apuestas_registradas"] == 0
    assert r1["n_apuestas_registradas"] > 0


def test_limite_de_apuestas_por_partido(cfg):
    """Limitar apuestas por partido acota la dependencia intra-evento."""
    run_daily("2026-09-12", cfg, ModelParams(max_bets_per_match=1), dry_run=True)
    d = db.fetch_df(cfg["storage"]["db_path"], "SELECT match_id, COUNT(*) n FROM bets GROUP BY match_id")
    assert (d["n"] <= 1).all()


def test_se_archivan_las_decisiones_rechazadas(cfg):
    """Sin el registro de rechazos no se puede caracterizar la seleccion muestral."""
    from pathlib import Path
    run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    logs = list(Path(cfg["storage"]["log_dir"]).glob("*.json"))
    assert logs
    decisiones = json.loads(logs[0].read_text(encoding="utf-8"))
    assert len(decisiones) >= 1
    assert {"aceptada", "motivo", "ev_calculado"} <= decisiones[0].keys()


def test_liquidacion_y_clv_end_to_end(cfg):
    dbp = cfg["storage"]["db_path"]
    run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    fila = db.fetch_df(dbp, "SELECT * FROM bets LIMIT 1").iloc[0]

    capturar_cierre(dbp, fila["match_id"], fila["mercado"], fila["linea"],
                    {"over": 1.70, "under": 2.15}, "run_test")
    liquidar_partido(dbp, fila["match_id"],
                     {"corners_totales": 12, "tarjetas_totales": 6})

    d = db.fetch_df(dbp, "SELECT * FROM bets WHERE id_apuesta = ?", [fila["id_apuesta"]])
    r = d.iloc[0]
    assert r["resultado_real"] in {"ganada", "perdida", "nula"}
    assert r["pnl_unidades"] is not None
    if r["seleccion"] == "over":
        # Se tomo 1.92 y cerro en 1.70: se batio la linea de cierre.
        assert r["clv_odds"] > 0
    # La cuota original permanece intacta tras la liquidacion.
    assert r["cuota_betplay"] == fila["cuota_betplay"]


def test_vista_roi_diario_tras_liquidar(cfg):
    dbp = cfg["storage"]["db_path"]
    run_daily("2026-09-12", cfg, ModelParams(), dry_run=True)
    mid = db.fetch_df(dbp, "SELECT match_id FROM bets LIMIT 1").iloc[0, 0]
    liquidar_partido(dbp, mid, {"corners_totales": 12, "tarjetas_totales": 6})
    v = db.fetch_df(dbp, "SELECT * FROM v_daily_roi")
    assert len(v) == 1 and v.loc[0, "n_apuestas"] >= 1
