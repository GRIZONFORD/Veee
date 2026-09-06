#!/usr/bin/env python3
"""Asistente de calibracion: descubre endpoints y selectores reales.

Los sitios no documentan su API interna y su DOM cambia sin aviso. Esta
herramienta reduce la calibracion a un par de iteraciones:

    # 1) Descargar y archivar una pagina para inspeccionarla
    python scripts/calibrate.py fetch https://www.example.com/laliga

    # 2) Ver el arbol de un JSON para deducir las rutas de config.yaml
    python scripts/calibrate.py tree data/raw/2026...html --max-depth 4

    # 3) Proponer selectores CSS buscando texto con forma de cuota decimal
    python scripts/calibrate.py selectors data/raw/2026...html

    # 4) Verificar que la configuracion ya calibrada extrae de verdad
    python scripts/calibrate.py verify --fuente betplay

El paso 3 localiza en el DOM los elementos cuyo texto parece una cuota decimal
(p. ej. 1.92 o 2,05) y reporta las rutas CSS mas frecuentes: en la practica, la
clase repetida que contiene esos valores es el selector `outcome_odds` buscado.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.config import load_config                                    # noqa: E402
from veee.scrapers.base import PoliteSession                           # noqa: E402

RE_CUOTA = re.compile(r"^\s*\d{1,3}[.,]\d{1,2}\s*$")


def cmd_fetch(a) -> int:
    s = PoliteSession(rate_limit_s=a.rate_limit, respect_robots=not a.ignore_robots)
    r = s.get(a.url)
    print(f"HTTP {r.status_code} · {len(r.text):,} bytes · {r.headers.get('content-type','?')}")
    destino = Path(a.out) if a.out else None
    if destino:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(r.text, encoding="utf-8")
        print(f"Guardado en {destino}")
    else:
        print("Archivado en data/raw/ (cache del cliente).")
    # Pista util: muchos sitios embeben su estado inicial como JSON en el HTML.
    for pista in ("__NEXT_DATA__", "__NUXT__", "window.__INITIAL_STATE__", "application/json"):
        if pista in r.text:
            print(f"  [PISTA] El HTML contiene '{pista}': probablemente haya un "
                  "JSON embebido mas estable que el DOM.")
    return 0


def _tree(obj, prof: int, max_prof: int, ruta: str = "") -> None:
    if prof > max_prof:
        return
    pad = "  " * prof
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:25]:
            tipo = type(v).__name__
            extra = f" ({len(v)})" if isinstance(v, (list, dict)) else ""
            muestra = "" if isinstance(v, (list, dict)) else f" = {str(v)[:60]!r}"
            print(f"{pad}{k}: {tipo}{extra}{muestra}")
            _tree(v, prof + 1, max_prof, f"{ruta}.{k}" if ruta else k)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0] de {len(obj)}:")
        _tree(obj[0], prof + 1, max_prof, f"{ruta}.0")


def cmd_tree(a) -> int:
    texto = Path(a.archivo).read_text(encoding="utf-8")
    try:
        data = json.loads(texto)
    except json.JSONDecodeError:
        # Intento de rescate: JSON embebido en un <script> de Next.js.
        m = re.search(r'__NEXT_DATA__[^>]*>(\{.*?\})</script>', texto, re.S)
        if not m:
            print("No es JSON y no se hallo JSON embebido. Use 'selectors'.")
            return 1
        data = json.loads(m.group(1))
        print("(JSON extraido de __NEXT_DATA__)\n")
    _tree(data, 0, a.max_depth)
    return 0


def _ruta_css(el) -> str:
    partes = []
    for p in list(el.parents)[:3][::-1]:
        if getattr(p, "name", None) in (None, "[document]"):
            continue
        cls = ".".join((p.get("class") or [])[:2])
        partes.append(f"{p.name}.{cls}" if cls else p.name)
    cls = ".".join((el.get("class") or [])[:2])
    partes.append(f"{el.name}.{cls}" if cls else el.name)
    return " > ".join(partes)


def cmd_selectors(a) -> int:
    soup = PoliteSession.soup(Path(a.archivo).read_text(encoding="utf-8"))
    cuotas = [el for el in soup.find_all(string=RE_CUOTA)]
    if not cuotas:
        print("No se hallo ningun texto con forma de cuota decimal.\n"
              "El sitio probablemente renderiza con JavaScript: busque el JSON\n"
              "embebido con 'tree', o use la API interna (pestana Red del navegador).")
        return 1
    print(f"{len(cuotas)} textos con forma de cuota encontrados.\n")
    rutas = Counter(_ruta_css(t.parent) for t in cuotas if t.parent)
    print("Rutas CSS candidatas para 'outcome_odds' (por frecuencia):")
    for ruta, n in rutas.most_common(8):
        print(f"  {n:>5}x  {ruta}")
    clases = Counter(c for t in cuotas if t.parent
                     for c in (t.parent.get("class") or []))
    if clases:
        print("\nClases mas frecuentes en los elementos de cuota:")
        for c, n in clases.most_common(8):
            print(f"  {n:>5}x  .{c}")
    print("\nLleve la mas repetida a config.yaml > betplay.selectors.outcome_odds")
    return 0


def cmd_verify(a) -> int:
    cfg = load_config(a.config)
    fecha = a.fecha or (date.today() + timedelta(days=1)).isoformat()
    if a.fuente == "betplay":
        from veee.scrapers.betplay import BetPlayScraper as S
    else:
        from veee.scrapers.linemate import LinemateScraper as S
    filas = S(cfg[a.fuente]).fetch(fecha)
    print(f"{a.fuente}: {len(filas)} filas para {fecha}")
    for f in filas[:8]:
        print("  " + json.dumps(f, ensure_ascii=False, default=str)[:150])
    if not filas:
        print("\n0 filas. Revise endpoint, selectores y el mapeo de mercados.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Asistente de calibracion de extractores.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Descarga y archiva una URL.")
    f.add_argument("url"); f.add_argument("--out")
    f.add_argument("--rate-limit", type=float, default=2.5)
    f.add_argument("--ignore-robots", action="store_true",
                   help="Solo para su propio sitio o con permiso explicito.")
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser("tree", help="Muestra el arbol de un JSON (o JSON embebido).")
    t.add_argument("archivo"); t.add_argument("--max-depth", type=int, default=3)
    t.set_defaults(func=cmd_tree)

    s = sub.add_parser("selectors", help="Propone selectores CSS a partir del HTML.")
    s.add_argument("archivo"); s.set_defaults(func=cmd_selectors)

    v = sub.add_parser("verify", help="Verifica que la configuracion extrae de verdad.")
    v.add_argument("--fuente", choices=["betplay", "linemate"], required=True)
    v.add_argument("--config", default="config/config.yaml"); v.add_argument("--fecha")
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
