#!/usr/bin/env python3
"""Estudio Monte Carlo de tamano y potencia del contraste (pre-registro).

Verifica que el procedimiento inferencial rechaza H0 aproximadamente el 5% de las
veces cuando el mercado es eficiente, y estima la potencia frente a alternativas.

Uso:
    python scripts/power_study.py --rep 200 --dias 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.econometrics import power_analysis                 # noqa: E402
from veee.model import ModelParams                           # noqa: E402
from veee.simulate import (escenario_h0, escenario_h1_ruido,  # noqa: E402
                           escenario_h1_subreaccion, monte_carlo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--dias", type=int, default=200)
    ap.add_argument("--out", default="outputs/potencia.json")
    a = ap.parse_args()

    params = ModelParams()
    escenarios = {
        "H0_eficiente": escenario_h0(n_dias=a.dias),
        "H1_subreaccion_lam0.6": escenario_h1_subreaccion(0.6, n_dias=a.dias),
        "H1_subreaccion_lam0.3": escenario_h1_subreaccion(0.3, n_dias=a.dias),
        "H1_ruido_nu0.30": escenario_h1_ruido(0.30, n_dias=a.dias),
    }
    res = {k: monte_carlo(v, params, n_rep=a.rep) for k, v in escenarios.items()}
    res["analitico"] = {f"roi_{r}": power_analysis(roi_objetivo=r)
                        for r in (0.01, 0.02, 0.03, 0.05)}

    print(f"{'escenario':<24}{'tasa rechazo':>14}{'ROI medio':>12}{'n medio':>10}")
    for k, v in res.items():
        if k == "analitico":
            continue
        print(f"{k:<24}{v['tasa_rechazo']:>14.3f}{v['roi_medio']:>12.4f}"
              f"{v['n_apuestas_medio']:>10.0f}")
    print("\nTamano muestral requerido (potencia 80%, alpha 5%):")
    for k, v in res["analitico"].items():
        print(f"  ROI objetivo {v['roi_objetivo']:.0%}: n >= {v['n_requerido']:,.0f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
