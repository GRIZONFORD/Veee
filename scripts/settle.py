#!/usr/bin/env python3
"""Liquidacion ex-post de partidos a partir de un JSON de resultados.

Formato esperado del archivo de resultados::

    [{"match_id": "abc123",
      "cierre": {"mercado": "corners_ou", "linea": 9.5,
                 "cuotas": {"over": 1.85, "under": 1.95}},
      "resultados": {"corners_totales": 11, "tarjetas_totales": 5,
                     "goles_local": 2, "goles_visitante": 1,
                     "resultado_1x2": "home"}}]

Uso:
    python scripts/settle.py --resultados data/resultados_2026-09-12.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee import database as db                              # noqa: E402
from veee.config import load_config                          # noqa: E402
from veee.settlement import capturar_cierre, liquidar_partido  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Liquidacion de apuestas y captura de cierre.")
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    cfg = load_config(a.config)
    db_path = cfg.get("storage", {}).get("db_path", str(db.DEFAULT_DB))
    partidos = json.loads(Path(a.resultados).read_text(encoding="utf-8"))
    run_id = db.new_run_id()

    total = 0
    for p in partidos:
        # 1) La linea de cierre DEBE registrarse antes de liquidar: el CLV es la
        #    metrica de mayor potencia del estudio y solo existe si se captura.
        if (cierre := p.get("cierre")):
            capturar_cierre(db_path, p["match_id"], cierre["mercado"],
                            cierre.get("linea"), cierre["cuotas"], run_id)
        total += liquidar_partido(db_path, p["match_id"], p["resultados"])

    print(json.dumps({"partidos": len(partidos), "apuestas_liquidadas": total},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
