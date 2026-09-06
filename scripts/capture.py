#!/usr/bin/env python3
"""Capturador continuo de cuotas, tendencias y lineas de cierre.

Disenado para ejecutarse por cron cada 5 minutos durante toda la temporada. Cada
invocacion es idempotente y de corta duracion: no mantiene estado en memoria, de
modo que un reinicio de la maquina no pierde nada.

    */5 * * * * cd /ruta/Veee && ./scripts/capture.py >> logs/capture.log 2>&1

Codigos de salida (para que cron o el runner alerten):
    0  ok
    1  error de ejecucion
    2  ALERTA de salud: el volumen capturado cayo respecto a la mediana movil
    3  CRITICO: cero filas capturadas; el extractor esta roto
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.capture import CaptureParams, run_capture_cycle                 # noqa: E402
from veee.config import load_config, model_params_from_cfg                # noqa: E402

SALIDA = {"ok": 0, "alerta": 2, "critico": 3}


def capture_params_from_cfg(cfg: dict) -> CaptureParams:
    c = cfg.get("captura", {}) or {}
    kw = {}
    if "ventanas_h" in c:
        kw["ventanas_h"] = tuple(float(x) for x in c["ventanas_h"])
    if "bet_window" in c:
        kw["bet_window"] = (float(c["bet_window"][0]), float(c["bet_window"][1]))
    for k in ("horizonte_h", "salud_umbral", "salud_min_muestras"):
        if k in c:
            kw[k] = c[k]
    return CaptureParams(**kw)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tick del capturador continuo.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Offline, con fixtures.")
    ap.add_argument("--ahora", help="Instante simulado ISO-8601 (pruebas).")
    ap.add_argument("--json", action="store_true", help="Solo JSON en stdout.")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else (logging.WARNING if a.json else logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    cfg = load_config(a.config)
    ahora = (datetime.fromisoformat(a.ahora.replace("Z", "+00:00")).astimezone(timezone.utc)
             if a.ahora else None)
    r = run_capture_cycle(cfg, model_params_from_cfg(cfg),
                          capture_params_from_cfg(cfg), dry_run=a.dry_run, ahora=ahora)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return SALIDA.get(r["estado"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
