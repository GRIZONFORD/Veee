#!/usr/bin/env python3
"""ClubElo: descarga historica, crosswalk de nombres y enriquecimiento de la Vía A.

    # 1) Probar conectividad antes de nada (falla en segundos, no en minutos)
    ./scripts/clubelo.py probar

    # 2) Construir el crosswalk de nombres archivo <-> ClubElo para una liga
    ./scripts/clubelo.py crosswalk --div SP1 --temporada 2024

    # 3) Descargar el historial completo de los equipos ya emparejados
    ./scripts/clubelo.py fetch --div SP1 --temporada 2024

    # 4) Enriquecer la Vía A con Elo a fecha exacta y ajustar el logit ordenado
    ./scripts/clubelo.py enrich --ligas SP1 --desde 2015 --hasta 2024

    # Sin red: todo el flujo sobre los fixtures ya incluidos en el repo
    ./scripts/clubelo.py crosswalk --fixtures
    ./scripts/clubelo.py enrich --fixtures

El paso 1 no es opcional: la escalada (crosswalk sobre N equipos, descarga de
N historiales) puede tardar minutos en fallar por fichero si el host esta
inalcanzable. Ver el mismo problema resuelto para el archivo historico en
scripts/archive.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                                   # noqa: E402

from veee import clubelo as ce                                        # noqa: E402
from veee.archive import etiqueta_temporada, load_season, load_many    # noqa: E402
from veee.archive_analysis import build_matrix                         # noqa: E402

FIXTURES_ARCHIVO = [("data/fixtures/archive/SP1_1011.csv", "SP1", "2010-11"),
                    ("data/fixtures/archive/SP1_2425.csv", "SP1", "2024-25")]
FIXTURE_SNAPSHOT = "data/fixtures/clubelo/snapshots/2024-08-17.csv"
FIXTURE_HISTORIAL_DIR = "data/fixtures/clubelo/historial"


def _cargar_archivo(a):
    if a.fixtures:
        return [load_season(p, d, t) for p, d, t in FIXTURES_ARCHIVO]
    divs = [x.strip() for x in a.ligas.split(",") if x.strip()] if hasattr(a, "ligas") \
        else [a.div]
    return load_many(divs, range(a.desde, a.hasta + 1), cache_dir=a.cache_archivo,
                     descargar_si_falta=not a.sin_descarga)


def cmd_probar(a) -> int:
    ok, detalle = ce.probar_conexion(cache_dir=a.cache, timeout=a.timeout)
    print(("[OK]    " if ok else "[FALLO] ") + detalle)
    if not ok:
        print("\napi.clubelo.com no responde. Aisle la causa antes de repetir un")
        print("barrido largo (mismo protocolo que scripts/archive.py):")
        print("  1. Abra en el navegador: http://api.clubelo.com/Barcelona")
        print("  2. En PowerShell: Test-NetConnection api.clubelo.com -Port 80")
        print("     (ClubElo sirve por HTTP simple, no HTTPS)")
    return 0 if ok else 1


def cmd_crosswalk(a) -> int:
    if a.fixtures:
        snap = ce.cargar_snapshot(FIXTURE_SNAPSHOT)
    else:
        ruta = ce.descargar_snapshot(a.fecha, cache_dir=a.cache, timeout=a.timeout,
                                     max_retries=a.retries)
        snap = ce.cargar_snapshot(ruta)

    cargadas = _cargar_archivo(a)
    nombres = sorted(set().union(*[
        set(c.partidos.equipo_local) | set(c.partidos.equipo_visitante) for c in cargadas
    ])) if cargadas else []
    if not nombres:
        print("Sin partidos del archivo (fixtures o descarga). Nada que emparejar.")
        return 1

    cw = ce.mapa_desde_snapshot(snap, nombres, pais=a.pais)
    print(f"{len(nombres)} equipos del archivo, snapshot ClubElo con "
          f"{snap['club'].nunique()} clubes ({a.pais or 'todos los paises'}).")
    print()
    print(cw[["origen", "destino", "metodo", "score", "requiere_revision"]]
         .to_string(index=False))
    n_rev = int(cw["requiere_revision"].sum())
    print(f"\nAutomaticos: {len(cw) - n_rev}/{len(cw)}  ·  a revisar: {n_rev}")
    if n_rev:
        print("Anada las correspondencias dudosas a ALIAS_MANUALES en crosswalk.py")
        print("antes de fiarse de 'enrich': un cruce erroneo no falla, produce un")
        print("Elo equivocado unido en silencio a un partido real.")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cw.to_csv(out / "clubelo_crosswalk.csv", index=False)
    print(f"\nGuardado en {out.resolve()}")
    return 0


def cmd_fetch(a) -> int:
    if a.fixtures:
        print("Modo --fixtures: no hay nada que descargar (ya estan en "
              f"{FIXTURE_HISTORIAL_DIR}).")
        return 0

    cw_path = Path(a.out_dir) / "clubelo_crosswalk.csv"
    if not cw_path.exists():
        print(f"Falta {cw_path}. Ejecute primero 'crosswalk'.")
        return 1
    import pandas as pd
    cw = pd.read_csv(cw_path)
    equipos = sorted(cw.loc[cw["destino"].notna(), "destino"].unique())
    if not equipos:
        print("El crosswalk no tiene ninguna correspondencia aceptada.")
        return 1

    print(f"Descargando el historial de {len(equipos)} equipos...")
    from veee.scrapers.base import PoliteSession
    sesion = PoliteSession(rate_limit_s=1.0, cache_dir=a.cache,
                           timeout=a.timeout, max_retries=a.retries)
    ok = fallos = 0
    try:
        for eq in equipos:
            try:
                ruta = ce.descargar_historial(eq, cache_dir=a.cache, session=sesion,
                                              forzar=a.forzar)
                print(f"  [OK]    {eq} -> {ruta}")
                ok += 1
            except Exception as exc:                     # noqa: BLE001
                print(f"  [FALLO] {eq}: {exc}")
                fallos += 1
    except KeyboardInterrupt:
        print(f"\nInterrumpido tras {ok} descargas ({fallos} fallidas). "
              "Lo ya cacheado no se repite al relanzar.")
        return 130
    print(f"\nDescargados {ok}, fallidos {fallos}.")
    return 0 if ok else 1


def cmd_enrich(a) -> int:
    cargadas = _cargar_archivo(a)
    if not cargadas:
        print("Sin partidos del archivo.")
        return 1
    matriz = build_matrix(cargadas, method=a.devig)
    nombres = sorted(set(matriz.equipo_local) | set(matriz.equipo_visitante))

    if a.fixtures:
        snap = ce.cargar_snapshot(FIXTURE_SNAPSHOT)
        historial_dir = FIXTURE_HISTORIAL_DIR
    else:
        snapshots = sorted(glob.glob(f"{a.cache}/snapshots/*.csv"))
        if not snapshots:
            print(f"Sin snapshot en {a.cache}/snapshots/. Ejecute 'crosswalk' primero "
                  "(descarga uno automaticamente).")
            return 1
        snap = ce.cargar_snapshot(snapshots[-1])
        historial_dir = f"{a.cache}/historial"

    historiales: dict[str, pd.DataFrame] = {}
    for path in glob.glob(f"{historial_dir}/*.csv"):
        try:
            h = ce.cargar_historial(path)
            historiales[h["club"].iloc[0]] = h
        except Exception as exc:                             # noqa: BLE001
            print(f"  [AVISO] {path}: {exc}")
    if not historiales:
        print(f"Sin historiales en {historial_dir}. Ejecute 'fetch' primero.")
        return 1

    cw = ce.mapa_desde_snapshot(snap, nombres, pais=a.pais)
    mapa = {r.origen: r.destino for r in cw.itertuples() if r.destino}
    if (n_rev := int(cw["requiere_revision"].sum())):
        print(f"AVISO: {n_rev} equipos del crosswalk sin resolver automaticamente; "
              "sus partidos quedaran sin cobertura de Elo. Ejecute 'crosswalk' para "
              "verlos y anadalos a ALIAS_MANUALES si corresponde.")

    partidos = (matriz[["match_id", "fecha", "equipo_local", "equipo_visitante",
                        "resultado_ft"]].drop_duplicates("match_id"))
    enr = ce.enriquecer_con_elo(partidos, historiales, mapa)

    n_cob = int(enr["elo_cobertura"].sum())
    print(f"Cobertura de Elo: {n_cob}/{len(enr)} partidos "
          f"({n_cob / max(len(enr), 1):.1%}).")
    if n_cob < len(enr):
        motivos = pd.concat([enr.loc[~enr.elo_cobertura, "elo_local_estado"],
                             enr.loc[~enr.elo_cobertura, "elo_visitante_estado"]])
        print("Motivos de falta de cobertura:")
        print(motivos[motivos != "ok"].value_counts().to_string())

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    enr.to_csv(out / "partidos_con_elo.csv", index=False)

    if n_cob < 10:
        print(f"\nSolo {n_cob} partidos con Elo y resultado: insuficiente para "
              "calibrar el logit ordenado (umbral: 10). Esto es lo esperado con "
              "--fixtures (7 partidos, valida la mecanica, no es evidencia).")
        print(f"\nGuardado en {out.resolve()}")
        return 0

    modelo = ce.ajustar_modelo_elo(enr)
    evaluacion = ce.evaluar_modelo_elo(modelo, enr)
    print("\n" + "=" * 74)
    print(f"LOGIT ORDENADO: resultado ~ elo_diff  (n={modelo.n}, "
          f"{modelo.rango_fechas[0]} a {modelo.rango_fechas[1]})")
    print("=" * 74)
    print(json.dumps(evaluacion, indent=2, ensure_ascii=False, default=str))

    (out / "modelo_elo_resumen.json").write_text(
        json.dumps({"n": modelo.n, "rango_fechas": modelo.rango_fechas,
                   **evaluacion}, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado en {out.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comunes(p):
        p.add_argument("--ligas", default="SP1")
        p.add_argument("--desde", type=int, default=2015)
        p.add_argument("--hasta", type=int, default=2024)
        p.add_argument("--cache-archivo", default="data/archive")
        p.add_argument("--cache", default="data/clubelo")
        p.add_argument("--out-dir", default="outputs/clubelo")
        p.add_argument("--fixtures", action="store_true",
                       help="Sin red: usa los fixtures del repo.")
        p.add_argument("--sin-descarga", action="store_true")
        p.add_argument("--timeout", type=int, default=10)
        p.add_argument("--retries", type=int, default=2)

    pr = sub.add_parser("probar", help="Prueba de conectividad rapida.")
    pr.add_argument("--cache", default="data/clubelo")
    pr.add_argument("--timeout", type=int, default=8)
    pr.set_defaults(func=cmd_probar)

    cw = sub.add_parser("crosswalk", help="Empareja nombres archivo <-> ClubElo.")
    comunes(cw)
    cw.add_argument("--fecha", default=None, help="ISO YYYY-MM-DD (def. hoy).")
    cw.add_argument("--pais", default="ESP")
    cw.set_defaults(func=cmd_crosswalk)

    fe = sub.add_parser("fetch", help="Descarga historiales ya emparejados.")
    comunes(fe)
    fe.add_argument("--forzar", action="store_true")
    fe.set_defaults(func=cmd_fetch)

    en = sub.add_parser("enrich", help="Enriquece la Via A y ajusta el logit ordenado.")
    comunes(en)
    en.add_argument("--devig", default="shin",
                    choices=["proportional", "additive", "power", "shin"])
    en.add_argument("--pais", default="ESP")
    en.set_defaults(func=cmd_enrich)

    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)-7s | %(message)s")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
