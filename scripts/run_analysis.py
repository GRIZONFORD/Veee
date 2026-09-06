#!/usr/bin/env python3
"""Analisis econometrico completo y generacion de tablas para el paper.

Uso:
    python scripts/run_analysis.py                     # sobre data/veee.db
    python scripts/run_analysis.py --csv matriz.csv    # sobre un CSV externo
    python scripts/run_analysis.py --simular h1_ruido  # sobre datos sinteticos
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee import database as db                              # noqa: E402
from veee.config import load_config, model_params_from_cfg   # noqa: E402
from veee.econometrics import (FORMULA_EFICIENCIA, calibration_table,  # noqa: E402
                               daily_roi, fit_logit, prepare, run_full_analysis)


def _cargar(a) -> pd.DataFrame:
    if a.simular:
        from veee.simulate import (escenario_h0, escenario_h1_ruido,
                                   escenario_h1_subreaccion, simular_muestra)
        esc = {"h0": escenario_h0, "h1_subreaccion": escenario_h1_subreaccion,
               "h1_ruido": escenario_h1_ruido}[a.simular](n_dias=a.dias)
        cfg = load_config(a.config) if Path(a.config).exists() else {}
        return simular_muestra(esc, model_params_from_cfg(cfg))
    if a.csv:
        return pd.read_csv(a.csv)
    cfg = load_config(a.config)
    return db.fetch_df(cfg.get("storage", {}).get("db_path", str(db.DEFAULT_DB)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inferencia econometrica del estudio.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--csv", help="Matriz econometrica en CSV.")
    ap.add_argument("--simular", choices=["h0", "h1_subreaccion", "h1_ruido"],
                    help="Ejecuta sobre datos sinteticos (validacion del diseno).")
    ap.add_argument("--dias", type=int, default=300)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    df = _cargar(a)
    if df.empty:
        print("No hay apuestas liquidadas todavia.")
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    informe = run_full_analysis(df)
    (out / "informe_econometrico.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    d = prepare(df)
    daily_roi(d).to_csv(out / "roi_diario.csv", index=False)
    try:
        res, dd = fit_logit(d, FORMULA_EFICIENCIA)
        (out / "logit_summary.txt").write_text(str(res.summary()), encoding="utf-8")
        calibration_table(res, dd).to_csv(out / "calibracion.csv", index=False)
    except Exception as exc:                                  # noqa: BLE001
        logging.warning("Logit no estimable: %s", exc)

    _imprimir(informe)
    print(f"\nSalidas escritas en {out.resolve()}")
    return 0


def _imprimir(inf: dict) -> None:
    d = inf["descriptivos"]
    print("\n" + "=" * 74)
    print("ESTADISTICOS DESCRIPTIVOS")
    print("=" * 74)
    for k, v in d.items():
        print(f"  {k:<24} {v:>14.4f}" if isinstance(v, float) else f"  {k:<24} {v:>14}")

    print("\n" + "=" * 74)
    print("CONTRASTES DE HIPOTESIS   (H0: mu <= 0  frente a  H1: mu > 0)")
    print("=" * 74)
    print(f"  {'contraste':<42}{'media':>10}{'estad.':>9}{'p-valor':>10}")
    for _, t in inf["contrastes"].items():
        est = t["estadistico"]
        print(f"  {t['nombre'][:41]:<42}{t['media']:>10.4f}{est:>9.3f}{t['p_valor']:>10.4f}"
              + ("  *" if t["p_valor"] < 0.05 else ""))

    print("\n" + "=" * 74)
    print("MODELO LOGIT")
    print("=" * 74)
    if (adv := inf.get("advertencia_especificacion")):
        import textwrap
        print(textwrap.fill("ADVERTENCIA: " + adv, 74, subsequent_indent="  "))
    for etiqueta, m in inf["logit"].items():
        if "error" in m:
            print(f"  [{etiqueta}] no estimable: {m['error']}")
            continue
        print(f"\n  [{etiqueta}] {m['formula']}   (n={m['n']})")
        print(f"    {'variable':<16}{'coef':>10}{'EE':>9}{'z':>8}{'p':>9}{'OR':>9}")
        for v in m["coeficientes"]:
            print(f"    {v:<16}{m['coeficientes'][v]:>10.4f}{m['ee_agrupados'][v]:>9.4f}"
                  f"{m['z'][v]:>8.2f}{m['p_valores'][v]:>9.4f}{m['odds_ratios'][v]:>9.3f}")
        g = m["diagnosticos"]
        print(f"    pseudo-R2={g['pseudo_R2_McFadden']:.4f}  AUC={g['AUC']:.4f}  "
              f"Brier={g['Brier']:.4f} (base {g['Brier_referencia_base']:.4f})  "
              f"HL p={g['HL_p_valor']:.3f}  linktest p={g['linktest_p_xb2']:.3f}")
        if (e := m.get("eficiencia")):
            if "beta_precio" in e:
                print(f"    Calibracion del precio: beta={e['beta_precio']:.3f} "
                      f"(H0: beta=1, p={e['p_valor_beta_precio_igual_1']:.4f}) "
                      f"-> {e['interpretacion']}")
            if "wald_p_valor" in e:
                print(f"    Wald conjunto {e['restricciones']}: "
                      f"chi2={e['wald_senal_conjunta']:.3f}, p={e['wald_p_valor']:.4f}")
                print(f"    => {e['conclusion']}")

    p = inf["potencia"]
    print("\n" + "=" * 74)
    print("ANALISIS DE POTENCIA")
    print("=" * 74)
    print(f"  Para detectar ROI={p['roi_objetivo']:.1%} con cuota media {p['cuota_media']:.2f} "
          f"(sigma={p['sigma_por_apuesta']:.3f}),")
    print(f"  potencia {p['potencia']:.0%} y alpha {p['alpha']:.2f}: "
          f"n >= {p['n_requerido']:,.0f} apuestas.")
    print(f"  Efecto minimo detectable con n=1.000: ROI = {p['mde_con_1000']:.2%}")

    if (pm := inf.get("por_mercado")):
        print("\n" + "=" * 74)
        print("HETEROGENEIDAD POR MERCADO (p ajustado por Benjamini-Hochberg)")
        print("=" * 74)
        print(f"  {'mercado':<16}{'n':>7}{'ROI':>10}{'t':>8}{'p':>9}{'p_BH':>9}")
        for f in pm:
            print(f"  {f['mercado']:<16}{f['n']:>7}{f['roi']:>10.4f}{f['t']:>8.2f}"
                  f"{f['p']:>9.4f}{f['p_BH']:>9.4f}" + ("  *" if f["significativo_BH"] else ""))


if __name__ == "__main__":
    raise SystemExit(main())
