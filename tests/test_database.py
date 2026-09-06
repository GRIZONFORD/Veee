"""Pruebas del esquema, la idempotencia y la proteccion anti look-ahead."""
import pytest

from veee import database as db


@pytest.fixture()
def base(tmp_path):
    p = tmp_path / "test.db"
    db.init_db(p)
    return p


def _fila_apuesta(bid="b1"):
    return {"id_apuesta": bid, "run_id": "r1", "fecha": "2026-09-12",
            "fecha_evento": "2026-09-12T19:00:00+00:00", "match_id": "m1",
            "partido": "A vs B", "mercado": "corners_ou", "seleccion": "over",
            "linea": 9.5, "cuota_betplay": 1.92, "p_implicita": 0.5208,
            "overround": 0.05, "p_mercado_devig": 0.496, "devig_method": "shin",
            "trend_score": 1.9, "p_estimada": 0.62, "ev_calculado": 0.19,
            "edge_prob": 0.12, "stake": 1.0, "model_version": "v1",
            "ts_colocacion": "2026-09-12T09:00:00+00:00"}


def test_id_apuesta_determinista():
    a = db.bet_id("m1", "corners_ou", "over", 9.5, "2026-09-12")
    b = db.bet_id("m1", "corners_ou", "over", 9.5, "2026-09-12")
    c = db.bet_id("m1", "corners_ou", "under", 9.5, "2026-09-12")
    assert a == b and a != c


def test_insercion_idempotente(base):
    with db.connect(base) as conn:
        db.upsert_match(conn, {"match_id": "m1", "temporada": "2025-2026",
                               "fecha_evento": "2026-09-12T19:00:00+00:00",
                               "equipo_local": "A", "equipo_visitante": "B"})
        assert db.insert_bet(conn, _fila_apuesta()) is True
        assert db.insert_bet(conn, _fila_apuesta()) is False   # no duplica
        n = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    assert n == 1


def test_bloqueo_de_reescritura_ex_ante(base):
    """El nucleo del protocolo: no se puede reescribir una decision ya tomada."""
    with db.connect(base) as conn:
        db.upsert_match(conn, {"match_id": "m1", "temporada": "2025-2026",
                               "fecha_evento": "2026-09-12T19:00:00+00:00",
                               "equipo_local": "A", "equipo_visitante": "B"})
        db.insert_bet(conn, _fila_apuesta())
        with pytest.raises(ValueError, match="look-ahead"):
            db.settle_bet(conn, "b1", cuota_betplay=5.00)
        with pytest.raises(ValueError, match="look-ahead"):
            db.settle_bet(conn, "b1", ev_calculado=0.99)
        # Los campos ex-post si son escribibles.
        db.settle_bet(conn, "b1", resultado_real="ganada", y_exito=1, pnl_unidades=0.92)
        r = conn.execute("SELECT resultado_real, cuota_betplay FROM bets").fetchone()
    assert r["resultado_real"] == "ganada" and r["cuota_betplay"] == 1.92


def test_restriccion_cuota_minima(base):
    import sqlite3
    with db.connect(base) as conn:
        db.upsert_match(conn, {"match_id": "m1", "temporada": "2025-2026",
                               "fecha_evento": "2026-09-12T19:00:00+00:00",
                               "equipo_local": "A", "equipo_visitante": "B"})
        fila = _fila_apuesta("b2") | {"cuota_betplay": 0.50}
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_bet(conn, fila)


def test_vista_roi_diario(base):
    with db.connect(base) as conn:
        db.upsert_match(conn, {"match_id": "m1", "temporada": "2025-2026",
                               "fecha_evento": "2026-09-12T19:00:00+00:00",
                               "equipo_local": "A", "equipo_visitante": "B"})
        db.insert_bet(conn, _fila_apuesta("b1"))
        db.settle_bet(conn, "b1", resultado_real="ganada", y_exito=1, pnl_unidades=0.92)
    d = db.fetch_df(base, "SELECT * FROM v_daily_roi")
    assert len(d) == 1 and d.loc[0, "roi_diario"] == pytest.approx(0.92)
