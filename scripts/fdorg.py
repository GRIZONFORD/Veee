#!/usr/bin/env python3
"""football-data.org: verificacion de la clave, calendario y emparejamiento.

La clave se lee SOLO del entorno; nunca se guarda en el repositorio:

    export FOOTBALL_DATA_ORG_TOKEN='su_clave'

    ./scripts/fdorg.py verify                       # ¿que cubre su plan?
    ./scripts/fdorg.py verify --competiciones PD,PL --temporadas 2022,2023,2024
    ./scripts/fdorg.py matches --competicion PD --temporada 2024
    ./scripts/fdorg.py crosswalk --competicion PD --temporada 2024 --div SP1

`verify` es el primer paso: en lugar de suponer que el plan gratuito cubre una
competicion o una temporada, lo COMPRUEBA y lo informa.

`crosswalk` construye el puente entre los nombres de football-data.org y los de
Football-Data.co.uk. Los emparejamientos dudosos se marcan para revision humana:
confundir dos clubes no produce ningun error, produce datos falsos.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.crosswalk import (emparejar_equipos, emparejar_partidos,  # noqa: E402
                            informe_emparejamiento, normalizar_nombre)
from veee.fdorg import (A_DIV_ARCHIVO, COMPETICIONES, FootballDataOrg,  # noqa: E402
                        FootballDataOrgError, normalizar_partidos, tabla_equipos)


def _cliente(a) -> FootballDataOrg:
    return FootballDataOrg(por_minuto=a.por_minuto)


def cmd_verify(a) -> int:
    comps = [c.strip() for c in a.competiciones.split(",") if c.strip()]
    temps = [int(t) for t in a.temporadas.split(",") if t.strip()]
    cli = _cliente(a)
    print("=" * 74)
    print("VERIFICACION DE LA CLAVE — que cubre realmente su plan")
    print("=" * 74)
    inf = cli.verificar(comps, temps)

    print(f"\nToken: {inf['token']}")
    vis = inf["competiciones_visibles"]
    print(f"Competiciones visibles: {len(vis)}")
    for c in vis[:20]:
        marca = " <-- objetivo" if c["code"] in comps else ""
        print(f"  {c['code']:5} {c['name']}{marca}")

    print("\nAcceso por competicion y temporada:")
    ok_total = 0
    for clave, r in inf["acceso"].items():
        if r.get("ok"):
            ok_total += 1
            print(f"  [OK]    {clave:12} {r['partidos']:>4} partidos "
                  f"({r.get('jugados', '?')} jugados)  {r.get('primero')} -> {r.get('ultimo')}")
            if r.get("trae_cuotas"):
                print("          incluye CUOTAS (paquete de pago activo)")
        else:
            print(f"  [DENEG] {clave:12} {r.get('motivo', '')[:90]}")

    print(f"\nPeticiones realizadas: {inf['peticiones_realizadas']}"
          f"  |  restantes este minuto: {inf['restantes_minuto']}")
    print("\nRECORDATORIO: football-data.org NO sirve cuotas en el plan gratuito.")
    print("Los precios del estudio salen de Football-Data.co.uk (scripts/archive.py).")
    print("Esta API aporta calendario con hora exacta, resultados e IDs estables")
    print("de equipo, que son el ancla del emparejamiento entre fuentes.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(inf, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nInforme en {a.out}")
    return 0 if ok_total else 1


def cmd_matches(a) -> int:
    cli = _cliente(a)
    df = normalizar_partidos(cli.partidos(a.competicion, a.temporada))
    if df.empty:
        print("Sin partidos.")
        return 1
    print(f"{len(df)} partidos · {COMPETICIONES.get(a.competicion, a.competicion)} "
          f"{a.temporada}")
    print(df["estado"].value_counts().to_string())
    print()
    print(df[["fecha", "jornada", "equipo_local", "equipo_visitante",
              "goles_local", "goles_visitante", "resultado_ft"]].head(15).to_string(index=False))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"fdorg_{a.competicion}_{a.temporada}.csv", index=False)
    tabla_equipos(df).to_csv(out / f"fdorg_equipos_{a.competicion}_{a.temporada}.csv",
                             index=False)
    print(f"\nGuardado en {out.resolve()}")
    return 0


def cmd_crosswalk(a) -> int:
    from veee.archive import etiqueta_temporada, load_season
    div = a.div or A_DIV_ARCHIVO.get(a.competicion)
    if not div:
        print(f"Sin equivalencia de division para {a.competicion}; use --div.")
        return 1

    cli = _cliente(a)
    fd = normalizar_partidos(cli.partidos(a.competicion, a.temporada))
    ruta = Path(a.cache) / f"{a.temporada % 100:02d}{(a.temporada + 1) % 100:02d}" / f"{div}.csv"
    if not ruta.exists():
        print(f"Falta el CSV del archivo: {ruta}\n"
              f"Ejecute: ./scripts/archive.py fetch --ligas {div} "
              f"--desde {a.temporada} --hasta {a.temporada}")
        return 1
    arch = load_season(ruta, div, etiqueta_temporada(a.temporada))

    nombres_fd = sorted(set(fd["equipo_local"]) | set(fd["equipo_visitante"]))
    nombres_ar = sorted(set(arch.partidos["equipo_local"]) |
                        set(arch.partidos["equipo_visitante"]))
    emp = emparejar_equipos(nombres_fd, nombres_ar)
    inf = informe_emparejamiento(emp)

    print("=" * 74)
    print(f"EMPAREJAMIENTO  football-data.org ({a.competicion}) <-> archivo ({div})")
    print("=" * 74)
    print(f"  equipos: {inf['n_equipos']}  ·  automaticos: {inf['automaticos']} "
          f"({inf['cobertura_automatica']:.0%})")
    print(f"  por metodo: {inf['por_metodo']}")

    if inf["a_revisar"]:
        print("\n  REQUIEREN REVISION HUMANA (no se resuelven por conjetura):")
        for r in inf["a_revisar"]:
            print(f"    {r['origen']:32} -> {str(r['destino']):24} "
                  f"[{r['metodo']}, {r['score']}]")
        print("\n  Anada las correspondencias a ALIAS_MANUALES en "
              "src/veee/crosswalk.py y vuelva a ejecutar.")

    mapa = {r["origen"]: normalizar_nombre(r["destino"])
            for _, r in emp.iterrows() if r["destino"]}
    fus = emparejar_partidos(fd, arch.partidos, mapa_equipos=mapa)
    cob = len(fus) / max(len(arch.partidos), 1)
    print(f"\n  partidos cruzados: {len(fus)} de {len(arch.partidos)} del archivo "
          f"({cob:.0%})")
    if cob < 0.9:
        print("  AVISO: cobertura baja. Revise los equipos marcados arriba antes")
        print("         de usar este cruce: un emparejamiento erroneo no falla,")
        print("         produce datos falsos.")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emp.to_csv(out / f"crosswalk_{a.competicion}_{div}_{a.temporada}.csv", index=False)
    print(f"\n  Guardado en {out.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--por-minuto", type=int, default=10,
                    help="Limite del plan gratuito (10/min).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Comprueba la clave y el alcance del plan.")
    v.add_argument("--competiciones", default="PD")
    v.add_argument("--temporadas", default="2024")
    v.add_argument("--out", default="outputs/fdorg_verificacion.json")
    v.set_defaults(func=cmd_verify)

    m = sub.add_parser("matches", help="Descarga el calendario y resultados.")
    m.add_argument("--competicion", default="PD")
    m.add_argument("--temporada", type=int, default=2024)
    m.add_argument("--out-dir", default="outputs/fdorg")
    m.set_defaults(func=cmd_matches)

    c = sub.add_parser("crosswalk", help="Puente de nombres con el archivo CSV.")
    c.add_argument("--competicion", default="PD")
    c.add_argument("--temporada", type=int, default=2024)
    c.add_argument("--div", help="Codigo de division del archivo (SP1, E0...).")
    c.add_argument("--cache", default="data/archive")
    c.add_argument("--out-dir", default="outputs/fdorg")
    c.set_defaults(func=cmd_crosswalk)

    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)-7s | %(message)s")
    try:
        return a.func(a)
    except FootballDataOrgError as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
