#!/usr/bin/env python3
"""Ciclo diario: extrae cuotas y tendencias, evalua +EV y registra apuestas ex-ante.

Uso:
    python scripts/run_daily.py --fecha 2026-09-12
    python scripts/run_daily.py --dry-run          # sin red, usa data/fixtures/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.config import load_config, model_params_from_cfg   # noqa: E402
from veee.pipeline import run_daily                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ciclo diario de paper trading.")
    ap.add_argument("--fecha", default=date.today().isoformat(), help="ISO YYYY-MM-DD")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Modo offline con fixtures: valida el contrato sin tocar la red.")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    cfg = load_config(a.config)
    resumen = run_daily(a.fecha, cfg, model_params_from_cfg(cfg), dry_run=a.dry_run)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
