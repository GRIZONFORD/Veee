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
                          descargar, load_many, load_season, probar_conexion,
                          resumen_muestra)
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

    n_total = (a.hasta - a.desde + 1) * len(divs)
    if not a.sin_preflight:
        print(f"Probando conectividad con football-data.co.uk antes de "
              f"descargar {n_total} ficheros...")
        ok_conn, detalle = probar_conexion(cache_dir=a.cache, timeout=a.timeout)
        if not ok_conn:
            print(f"\n  [FALLO] {detalle}\n")
            print("No se puede llegar a www.football-data.co.uk. Antes de repetir")
            print("un barrido largo, aisle la causa fuera de Python:")
            print()
            print("  1. Abra en el navegador (prueba mas simple y concluyente):")
            print("     https://www.football-data.co.uk/mmz4281/2425/E0.csv")
            print("     - Si SI descarga en el navegador pero falla aqui, algo del")
            print("       propio proceso Python esta bloqueado (antivirus/EDR) o")
            print("       necesita el proxy corporativo que el navegador ya usa.")
            print("     - Si TAMPOCO descarga en el navegador, el bloqueo es de")
            print("       red (firewall, ISP, o el sitio esta caido), no del script.")
            print()
            print("  2. En PowerShell, para separar DNS de la conexion TCP:")
            print("     Test-NetConnection www.football-data.co.uk -Port 443")
            print()
            print("  3. Si hay VPN o red corporativa/universitaria, pruebe")
            print("     desactivandola o desde otra red (p. ej. datos moviles).")
            print()
            print("  4. Un WinError 10060 (tiempo agotado, sin respuesta) en Windows")
            print("     a veces se debe a una ruta IPv6 rota que cuelga antes de")
            print("     caer a IPv4. Pruebe deshabilitando IPv6 en el adaptador de")
            print("     red, o fuerce IPv4 en su cliente VPN si usa uno.")
            print()
            print("Repita 'fetch' cuando el paso 1 o 2 confirmen conectividad. Para")
            print("omitir esta comprobacion: --sin-preflight")
            return 1
        print(f"  [OK] {detalle}\n")

    ok = fallos = 0
    try:
        for anio in range(a.desde, a.hasta + 1):
            for div in divs:
                try:
                    ruta = descargar(div, anio, cache_dir=a.cache, forzar=a.forzar,
                                     timeout=a.timeout, max_retries=a.retries)
                    print(f"  [OK]    {div} {etiqueta_temporada(anio)} -> {ruta}")
                    ok += 1
                except Exception as exc:                 # noqa: BLE001
                    # Que falte una combinacion liga-temporada es normal
                    # (ascensos, cambios de cobertura). Se registra y se sigue.
                    print(f"  [FALLO] {div} {etiqueta_temporada(anio)}: {exc}")
                    fallos += 1
    except KeyboardInterrupt:
        print(f"\nInterrumpido por el usuario tras {ok} descargas ({fallos} fallidas).")
        print("Los ficheros ya descargados quedan en cache: relance 'fetch' para")
        print("continuar donde quedo (no repite lo que ya existe en disco).")
        return 130
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
    f.add_argument("--timeout", type=int, default=10,
                   help="Segundos antes de dar por fallida una conexion (def. 10).")
    f.add_argument("--retries", type=int, default=2,
                   help="Reintentos por fichero antes de saltarlo (def. 2).")
    f.add_argument("--sin-preflight", action="store_true",
                   help="Omite la prueba de conectividad previa al barrido.")
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
