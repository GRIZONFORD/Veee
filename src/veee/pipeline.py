"""
pipeline.py — Orquestacion del ciclo diario de recoleccion y registro ex-ante.

Flujo (idempotente y reproducible)
----------------------------------
    1. Extraer cuotas de BetPlay para la fecha objetivo  -> odds_snapshots
    2. Extraer tendencias de Linemate                    -> trends
    3. Emparejar por `match_key` (liga|fecha|local|visitante canonizados)
    4. Neutralizar el margen y estimar p_est             -> model.build_candidate
    5. Filtrar +EV y registrar la apuesta EX-ANTE        -> bets
    6. Exportar el diario de decisiones (auditoria)

Cada ejecucion recibe un `run_id`; los IDs de apuesta son deterministas, de modo
que reejecutar el pipeline el mismo dia no duplica registros ni permite reescribir
decisiones pasadas (proteccion contra look-ahead).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import database as db
from .model import ModelParams, build_candidate
from .scrapers.betplay import BetPlayScraper
from .scrapers.linemate import LinemateScraper, trend_z

log = logging.getLogger(__name__)


def _match_row(o: dict[str, Any], temporada: str) -> dict[str, Any]:
    return {
        "match_id": o["match_key"], "liga": o.get("liga", "LaLiga"),
        "temporada": temporada, "fecha_evento": o["fecha_evento"],
        "equipo_local": o["equipo_local"], "equipo_visitante": o["equipo_visitante"],
    }


def _market_key(row: dict[str, Any]) -> tuple:
    """Un 'mercado' agrupa las selecciones exhaustivas y excluyentes a de-vigar."""
    return (row["match_key"], row["mercado"], row.get("linea"))


def run_daily(fecha: str, cfg: dict[str, Any], params: ModelParams,
              dry_run: bool = False) -> dict[str, Any]:
    """Ejecuta el ciclo completo para `fecha` (ISO YYYY-MM-DD). Devuelve un resumen."""
    db_path = cfg.get("storage", {}).get("db_path", str(db.DEFAULT_DB))
    temporada = cfg.get("estudio", {}).get("temporada", "2025-2026")
    run_id = db.new_run_id()
    db.init_db(db_path)

    fx = cfg.get("fixtures", {}) if dry_run else {}
    odds = BetPlayScraper(cfg["betplay"], fixture=fx.get("betplay")).fetch(fecha)
    trends = LinemateScraper(cfg["linemate"], fixture=fx.get("linemate")).fetch(fecha)

    # Indexacion de tendencias por (partido, mercado, seleccion, linea).
    tmap: dict[tuple, dict[str, Any]] = {
        (t["match_key"], t["mercado"], t["seleccion"], t.get("linea")): t for t in trends
    }
    # Agrupacion de cuotas por mercado para poder neutralizar el margen.
    grupos: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for o in odds:
        grupos[_market_key(o)].append(o)

    decisiones: list[dict[str, Any]] = []
    n_bets = 0
    por_partido: dict[str, int] = defaultdict(int)

    with db.connect(db_path) as conn:
        conn.execute("BEGIN")
        for o in odds:
            db.upsert_match(conn, _match_row(o, temporada))
            db.insert_snapshot(conn, {
                "match_id": o["match_key"], "casa": o.get("casa", "BetPlay"),
                "mercado": o["mercado"], "seleccion": o["seleccion"],
                "linea": o.get("linea"), "cuota": o["cuota"],
                "ts_captura": o["ts_captura"], "horas_al_inicio": o.get("horas_al_inicio"),
                "es_cierre": 0, "run_id": run_id,
            })

        for t in trends:
            db.insert_trend(conn, {
                "match_id": t["match_key"], "fuente": t.get("fuente", "Linemate"),
                "mercado": t["mercado"], "seleccion": t["seleccion"],
                "linea": t.get("linea"), "hits": t.get("hits"),
                "n_muestra": t.get("n_muestra"), "tasa_bruta": t.get("tasa_bruta"),
                "media_reciente": t.get("media_reciente"),
                "varianza_reciente": t.get("varianza_reciente"),
                "trend_score": None, "ts_captura": t["ts_captura"], "run_id": run_id,
            })

        for clave, selecciones in grupos.items():
            cuotas = [s["cuota"] for s in selecciones]
            if len(cuotas) < 2:
                continue                      # mercado incompleto: no se puede de-vigar
            for idx, sel in enumerate(selecciones):
                t = tmap.get((sel["match_key"], sel["mercado"], sel["seleccion"],
                              sel.get("linea")))
                if t is None:
                    continue                  # sin senal de Linemate -> no se evalua
                from .oddsmath import devig
                p_mkt = float(devig(cuotas, method=params.devig_method)[idx])
                z = trend_z(float(t["tasa_bruta"]), int(t["n_muestra"]), p_mkt)
                cand = build_candidate(t, sel["cuota"], cuotas, idx, z, params)
                registro = {
                    "partido": f"{sel['equipo_local']} vs {sel['equipo_visitante']}",
                    "mercado": sel["mercado"], "seleccion": sel["seleccion"],
                    "linea": sel.get("linea"), "cuota": sel["cuota"],
                    **{k: v for k, v in cand.items() if k != "diagnostico"},
                }
                decisiones.append(registro)

                if not cand["aceptada"]:
                    continue
                if por_partido[sel["match_key"]] >= params.max_bets_per_match:
                    registro["motivo"] = "limite_por_partido"
                    continue

                fecha_col = datetime.now(timezone.utc).date().isoformat()
                bid = db.bet_id(sel["match_key"], sel["mercado"], sel["seleccion"],
                                sel.get("linea"), fecha_col)
                inserted = db.insert_bet(conn, {
                    "id_apuesta": bid, "run_id": run_id, "fecha": fecha_col,
                    "fecha_evento": sel["fecha_evento"], "match_id": sel["match_key"],
                    "partido": registro["partido"], "mercado": sel["mercado"],
                    "seleccion": sel["seleccion"], "linea": sel.get("linea"),
                    "casa": sel.get("casa", "BetPlay"), "cuota_betplay": sel["cuota"],
                    "p_implicita": cand["p_implicita"], "overround": cand["overround"],
                    "p_mercado_devig": cand["p_mercado_devig"],
                    "devig_method": cand["devig_method"],
                    "trend_score": cand["trend_score"], "p_estimada": cand["p_estimada"],
                    "ev_calculado": cand["ev_calculado"], "edge_prob": cand["edge_prob"],
                    "stake": cfg.get("estudio", {}).get("stake", 1.0),
                    "kelly_sugerido": cand["kelly_sugerido"],
                    "model_version": cand["model_version"],
                    "ts_colocacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "horas_al_inicio": sel.get("horas_al_inicio"),
                })
                if inserted:
                    n_bets += 1
                    por_partido[sel["match_key"]] += 1
        conn.execute("COMMIT")

    resumen = {
        "run_id": run_id, "fecha": fecha, "n_cuotas": len(odds),
        "n_tendencias": len(trends), "n_evaluadas": len(decisiones),
        "n_apuestas_registradas": n_bets,
        "tasa_aceptacion": round(n_bets / len(decisiones), 4) if decisiones else 0.0,
    }
    _log_decisiones(decisiones, run_id, cfg)
    log.info("Resumen del ciclo: %s", resumen)
    return resumen


def _log_decisiones(decisiones: list[dict[str, Any]], run_id: str,
                    cfg: dict[str, Any]) -> Path | None:
    """Archiva TODAS las evaluaciones, aceptadas y rechazadas.

    Guardar los rechazos es indispensable: sin ellos no se puede caracterizar el
    proceso de seleccion muestral ni contrastar sesgos de seleccion en el paper.
    """
    if not decisiones:
        return None
    out = Path(cfg.get("storage", {}).get("log_dir", "outputs/decisiones"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    path.write_text(json.dumps(decisiones, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path
