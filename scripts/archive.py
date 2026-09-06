#!/usr/bin/env python3
"""Archivo historico de Football-Data.co.uk: descarga, cobertura y analisis.

    # 1) Descargar (gratis, sin clave). Cachea: no repide lo ya descargado.
    ./scripts/archive.py fetch --ligas SP1,E0,D1,I1,F1 --desde 2015 --hasta 2024

    # 2) VERIFICAR EL ESQUEMA antes de analizar nada
    ./scripts/archive.py report

    # 3) Preguntas A1-A5
    ./scripts/archive.py analyze

    # Sin red: sobre los fixtures que reproducen las dos epocas del esquema
    ./scripts/archive.py report --fixtures
    ./scripts/archive.py analyze --fixtures

El paso 2 no es opcional. El esquema de estos CSV cambia entre temporadas
(columnas de cierre ausentes en anos antiguos, agregados renombrados de
BbMx/BbAv a Max/Avg), y el informe de cobertura es el mecanismo que convierte
esa deriva en algo visible en vez de en muestra perdida en silencio.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.archive import (LIGAS, coverage_report, etiqueta_temporada,  # noqa: E402
                          descargar, load_many, load_season, resumen_muestra)
from veee.archive_analysis import (a1_sesgo_favorito_longshot,  # noqa: E402
                                   a2_asimetria_entre_casas, a3_clv_por_casa,
                                   a5_evolucion_por_temporada, build_matrix,
                                   desplazamiento_linea, resumen_matriz)

FIXTURES = [("data/fixtures/archive/SP1_1011.csv", "SP1", "2010-11"),
            ("data/fixtures/archive/SP1_2425.csv", "SP1", "2024-25")]


def _cargar(a):
    if a.fixtures:
        return [load_season(p, d, t) for p, d, t in FIXTURES]
    divs = [x.strip() for x in a.ligas.split(",") if x.strip()]
    return load_many(divs, range(a.desde, a.hasta + 1), cache_dir=a.cache,
                     descargar_si_falta=not a.sin_descarga)


# --------------------------------------------------------------------- fetch --
def cmd_fetch(a) -> int:
    divs = [x.strip() for x in a.ligas.split(",") if x.strip()]
    if desconocidas := [d for d in divs if d not in LIGAS]:
        print(f"Codigos de liga desconocidos: {desconocidas}")
        print(f"Disponibles: {', '.join(sorted(LIGAS))}")
        return 1
    ok = fallos = 0
    for anio in range(a.desde, a.hasta + 1):
        for div in divs:
            try:
                ruta = descargar(div, anio, cache_dir=a.cache, forzar=a.forzar)
                print(f"  [OK]    {div} {etiqueta_temporada(anio)} -> {ruta}")
                ok += 1
            except Exception as exc:                     # noqa: BLE001
                # Que falte una combinacion liga-temporada es normal (ascensos,
                # cambios de cobertura). Se registra y se sigue.
                print(f"  [FALLO] {div} {etiqueta_temporada(anio)}: {exc}")
                fallos += 1
    print(f"\nDescargados {ok}, fallidos {fallos}.")
    return 0 if ok else 1


# -------------------------------------------------------------------- report --
def cmd_report(a) -> int:
    cargadas = _cargar(a)
    if not cargadas:
        print("Sin datos. Ejecute primero 'fetch'.")
        return 1

    print("=" * 78)
    print("COBERTURA DEL ARCHIVO — verificacion del esquema")
    print("=" * 78)
    rep = coverage_report(cargadas)
    resumen = resumen_muestra(cargadas)

    print("\nTamano muestral efectivo:")
    for k, v in resumen.items():
        print(f"  {k:24} {v}")

    print("\nDisponibilidad de cuotas de CIERRE por temporada")
    print("  (el CLV solo puede calcularse donde exista)")
    piv = rep.pivot_table(index="temporada", values=["cierre", "apertura"],
                          aggfunc="sum")
    print(piv.to_string())

    print("\nCasas por temporada (x = presente, C = tambien con cierre)")
    tab = rep.assign(marca=lambda r: r["cierre"].map({True: "C", False: "x"}))
    print(tab.pivot_table(index="casa", columns="temporada", values="marca",
                          aggfunc="first").fillna("-").to_string())

    avisos = [(c.div, c.temporada, m) for c in cargadas for m in c.avisos]
    if avisos:
        print("\nAVISOS DE INTEGRIDAD:")
        for div, temp, m in avisos:
            print(f"  [{div} {temp}] {m}")
    else:
        print("\nSin avisos de integridad.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(a.out, index=False)
        print(f"\nInforme detallado en {a.out}")
    return 0


# ------------------------------------------------------------------- analyze --
def cmd_analyze(a) -> int:
    cargadas = _cargar(a)
    if not cargadas:
        print("Sin datos. Ejecute primero 'fetch'.")
        return 1
    d = build_matrix(cargadas, method=a.devig)
    res = resumen_matriz(d)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "matriz_archivo.csv", index=False)

    print("=" * 78)
    print(f"MATRIZ DEL ARCHIVO  (de-vig: {a.devig})")
    print("=" * 78)
    for k, v in res.items():
        print(f"  {k:20} {v}")
    if res["n_partidos"] < 500:
        print("\n  AVISO: muestra pequena. Las cifras siguientes verifican la")
        print("         mecanica, no constituyen evidencia econometrica.")

    print("\n" + "=" * 78)
    print("A1 — SESGO FAVORITO-LONGSHOT  (ROI por decil de probabilidad)")
    print("=" * 78)
    a1 = a1_sesgo_favorito_longshot(d)
    if not a1.empty:
        print(a1.round(4).to_string(index=False))
        a1.to_csv(out / "a1_favorito_longshot.csv", index=False)

    print("\n" + "=" * 78)
    print("A2 — ASIMETRIA ENTRE CASAS   H0: beta(desv)=0 y beta(sharp)=1")
    print("=" * 78)
    try:
        r, dd = a2_asimetria_entre_casas(d)
        from veee.econometrics import test_calibracion_precio
        print(f"  {'variable':<18}{'coef':>10}{'EE':>9}{'z':>8}{'p':>9}")
        for v in r.params.index:
            print(f"  {v:<18}{r.params[v]:>10.4f}{r.bse[v]:>9.4f}"
                  f"{r.tvalues[v]:>8.2f}{r.pvalues[v]:>9.4f}")
        print(f"  n={len(dd)}  pseudo-R2={r.prsquared:.4f}")
        if "desv_pct" in r.pvalues:
            sig = r.pvalues["desv_pct"] < 0.05
            print(f"\n  => {'SE RECHAZA' if sig else 'NO se rechaza'} la eficiencia: "
                  f"la discrepancia entre casas {'SI' if sig else 'no'} predice "
                  "el resultado condicionando sobre el precio de referencia.")
        (out / "a2_logit.txt").write_text(str(r.summary()), encoding="utf-8")
    except Exception as exc:                             # noqa: BLE001
        print(f"  No estimable: {exc}")

    print("\n" + "=" * 78)
    print("A3 — CLV POR CASA  (frente a la referencia sharp)")
    print("=" * 78)
    a3 = a3_clv_por_casa(d)
    if not a3.empty:
        print(a3.round(4).to_string(index=False))
        a3.to_csv(out / "a3_clv_por_casa.csv", index=False)

    print("\n" + "=" * 78)
    print("A4 — DESPLAZAMIENTO APERTURA -> CIERRE")
    print("=" * 78)
    a4 = desplazamiento_linea(d)
    if a4.empty:
        print("  Sin temporadas con ambas fases: el archivo antiguo no trae cierre.")
    else:
        print(a4.groupby("casa")[["drift_p", "drift_cuota", "clv_prob"]]
                .agg(["mean", "size"]).round(4).to_string())
        a4.to_csv(out / "a4_desplazamiento.csv", index=False)

    print("\n" + "=" * 78)
    print("A5 — EVOLUCION POR TEMPORADA")
    print("=" * 78)
    a5 = a5_evolucion_por_temporada(d)
    print(a5.round(4).to_string(index=False))
    a5.to_csv(out / "a5_evolucion.csv", index=False)

    (out / "resumen_archivo.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSalidas en {out.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comunes(p, con_red=True):
        p.add_argument("--ligas", default="SP1")
        p.add_argument("--desde", type=int, default=2015)
        p.add_argument("--hasta", type=int, default=2024)
        p.add_argument("--cache", default="data/archive")
        if con_red:
            p.add_argument("--fixtures", action="store_true",
                           help="Sin red: usa los fixtures de las dos epocas.")
            p.add_argument("--sin-descarga", action="store_true")

    f = sub.add_parser("fetch", help="Descarga los CSV al cache local.")
    comunes(f, con_red=False)
    f.add_argument("--forzar", action="store_true")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("report", help="Informe de cobertura y verificacion de esquema.")
    comunes(r)
    r.add_argument("--out", default="outputs/cobertura_archivo.csv")
    r.set_defaults(func=cmd_report)

    an = sub.add_parser("analyze", help="Preguntas A1-A5.")
    comunes(an)
    an.add_argument("--devig", default="shin",
                    choices=["proportional", "additive", "power", "shin"])
    an.add_argument("--out-dir", default="outputs/archivo")
    an.set_defaults(func=cmd_analyze)

    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)-7s | %(message)s")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
