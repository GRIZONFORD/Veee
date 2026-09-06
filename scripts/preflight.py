#!/usr/bin/env python3
"""Validacion previa al despliegue. Ejecutar ANTES de activar el cron.

Comprueba, en orden, todo lo que puede impedir una recoleccion valida:

  1. La configuracion no contiene marcadores de posicion sin calibrar.
  2. Los dominios son alcanzables (DNS, TCP, TLS).
  3. `robots.txt` permite el acceso automatizado a las rutas que se van a usar.
  4. Una captura real devuelve datos que cumplen el contrato de campos y rangos.
  5. **Las claves de partido de BetPlay y Linemate cruzan entre si.**

La comprobacion 5 es la que mas probablemente falle en silencio: si la
canonizacion de nombres de equipo no coincide entre fuentes, el cruce da cero,
no se registra ninguna apuesta y el sistema parece funcionar con normalidad
durante semanas. Se comprueba explicitamente y se listan los nombres huerfanos.

    python scripts/preflight.py            # validacion completa (usa la red)
    python scripts/preflight.py --offline  # solo configuracion y contrato

Salida: 0 si todo pasa (GO); 1 si hay algun bloqueo (NO-GO).
"""
from __future__ import annotations

import argparse
import logging
import socket
import ssl
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veee.config import load_config                                    # noqa: E402
from veee.oddsmath import MAX_ODDS, MIN_ODDS                           # noqa: E402
from veee.scrapers.base import USER_AGENT                              # noqa: E402
from veee.scrapers.betplay import BetPlayScraper                       # noqa: E402
from veee.scrapers.linemate import LinemateScraper                     # noqa: E402

MARCADORES = ("ENDPOINT.A.CALIBRAR", "A.CALIBRAR", "CAMBIAR", "TODO", "XXX")
CAMPOS_ODDS = {"match_key", "mercado", "seleccion", "cuota", "fecha_evento",
               "equipo_local", "equipo_visitante"}
CAMPOS_TREND = {"match_key", "mercado", "seleccion", "hits", "n_muestra", "tasa_bruta"}


class Reporte:
    def __init__(self) -> None:
        self.fallos: list[str] = []
        self.avisos: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"  [OK]     {msg}")

    def aviso(self, msg: str) -> None:
        self.avisos.append(msg)
        print(f"  [AVISO]  {msg}")

    def fallo(self, msg: str) -> None:
        self.fallos.append(msg)
        print(f"  [FALLO]  {msg}")


# --------------------------------------------------------------------------- #
def check_config(cfg: dict, r: Reporte) -> None:
    print("\n1. Configuracion")
    def _buscar(obj, ruta=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _buscar(v, f"{ruta}.{k}" if ruta else k)
        elif isinstance(obj, str):
            for m in MARCADORES:
                if m in obj:
                    yield ruta, obj
                    break
    pendientes = list(_buscar(cfg))
    if pendientes:
        for ruta, val in pendientes:
            r.fallo(f"Sin calibrar: {ruta} = {val!r}")
    else:
        r.ok("Sin marcadores de posicion pendientes.")

    for bloque in ("betplay", "linemate"):
        c = cfg.get(bloque, {})
        modo = c.get("mode", "api")
        clave = "api_url" if modo == "api" else "html_url"
        if not c.get(clave):
            r.fallo(f"{bloque}: falta '{clave}' para mode='{modo}'.")
        else:
            r.ok(f"{bloque}: mode='{modo}', {clave} definido.")
        if c.get("rate_limit_s", 0) < 1.0:
            r.aviso(f"{bloque}: rate_limit_s < 1s. Suba el intervalo por cortesia.")
        if not c.get("respect_robots", True):
            r.aviso(f"{bloque}: respect_robots esta desactivado.")

    cap = cfg.get("captura", {})
    if cap:
        bw = cap.get("bet_window", [24, 6])
        if not (bw[0] > bw[1] >= 0):
            r.fallo(f"captura.bet_window mal definida: {bw} (debe ser [desde, hasta]).")
        else:
            r.ok(f"Ventana de colocacion pre-registrada: T-{bw[0]}h a T-{bw[1]}h.")
    else:
        r.aviso("Sin bloque 'captura': se usaran los valores por defecto.")


def check_red(cfg: dict, r: Reporte) -> None:
    print("\n2. Alcanzabilidad de red (DNS / TCP / TLS)")
    for bloque in ("betplay", "linemate"):
        c = cfg.get(bloque, {})
        url = c.get("api_url") or c.get("html_url")
        if not url:
            continue
        host = urlparse(url).netloc.split(":")[0]
        if not host or "CALIBRAR" in host:
            r.fallo(f"{bloque}: host no calibrado.")
            continue
        try:
            ip = socket.gethostbyname(host)
        except OSError as exc:
            r.fallo(f"{bloque}: DNS falla para {host} ({exc}).")
            continue
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=10) as s:
                with ctx.wrap_socket(s, server_hostname=host):
                    pass
            r.ok(f"{bloque}: {host} -> {ip}, TLS correcto.")
        except Exception as exc:                                        # noqa: BLE001
            r.fallo(f"{bloque}: no se pudo abrir TLS con {host} ({exc}). "
                    "Puede ser bloqueo geografico o de politica de red.")


def check_robots(cfg: dict, r: Reporte) -> None:
    print("\n3. robots.txt")
    for bloque in ("betplay", "linemate"):
        c = cfg.get(bloque, {})
        url = c.get("api_url") or c.get("html_url")
        if not url or "CALIBRAR" in url:
            continue
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception as exc:                                        # noqa: BLE001
            r.aviso(f"{bloque}: robots.txt inaccesible ({exc}). Revise los ToS a mano.")
            continue
        if rp.can_fetch(USER_AGENT, url):
            r.ok(f"{bloque}: robots.txt permite {url}")
        else:
            r.fallo(f"{bloque}: robots.txt PROHIBE {url}. No proceda con la captura.")


def check_contrato(cfg: dict, r: Reporte, offline: bool) -> tuple[list, list]:
    print("\n4. Contrato de datos")
    fx = cfg.get("fixtures", {}) if offline else {}
    fecha = (date.today() + timedelta(days=1)).isoformat()
    odds: list = []
    trends: list = []
    try:
        odds = BetPlayScraper(cfg["betplay"], fixture=fx.get("betplay")).fetch(fecha)
    except Exception as exc:                                            # noqa: BLE001
        r.fallo(f"BetPlay: la extraccion fallo ({type(exc).__name__}: {exc}).")
    try:
        trends = LinemateScraper(cfg["linemate"], fixture=fx.get("linemate")).fetch(fecha)
    except Exception as exc:                                            # noqa: BLE001
        r.fallo(f"Linemate: la extraccion fallo ({type(exc).__name__}: {exc}).")

    if not odds:
        r.fallo("BetPlay devolvio 0 selecciones: selectores o endpoint incorrectos.")
    else:
        faltan = CAMPOS_ODDS - set(odds[0])
        if faltan:
            r.fallo(f"BetPlay: faltan campos {sorted(faltan)}.")
        else:
            r.ok(f"BetPlay: {len(odds)} selecciones con el contrato completo.")
        malas = [o["cuota"] for o in odds if not (MIN_ODDS <= o["cuota"] <= MAX_ODDS)]
        if malas:
            r.fallo(f"BetPlay: {len(malas)} cuotas fuera de rango (ej. {malas[:3]}).")
        else:
            r.ok("BetPlay: todas las cuotas en rango admisible.")
        # Un mercado necesita >= 2 selecciones para poder neutralizar el margen.
        from collections import Counter
        grupos = Counter((o["match_key"], o["mercado"], o.get("linea")) for o in odds)
        sueltos = sum(1 for v in grupos.values() if v < 2)
        if sueltos:
            r.aviso(f"{sueltos} mercados con una sola seleccion: no se podra de-vigar.")
        else:
            r.ok(f"{len(grupos)} mercados completos para neutralizar el margen.")

    if not trends:
        r.fallo("Linemate devolvio 0 tendencias: selectores o endpoint incorrectos.")
    else:
        faltan = CAMPOS_TREND - set(trends[0])
        if faltan:
            r.fallo(f"Linemate: faltan campos {sorted(faltan)}.")
        else:
            r.ok(f"Linemate: {len(trends)} tendencias con el contrato completo.")
    return odds, trends


def check_cruce(odds: list, trends: list, r: Reporte) -> None:
    print("\n5. Cruce BetPlay <-> Linemate  (fallo silencioso mas probable)")
    if not odds or not trends:
        r.fallo("No se puede validar el cruce sin datos de ambas fuentes.")
        return
    k_odds = {o["match_key"] for o in odds}
    k_tr = {t["match_key"] for t in trends}
    comunes = k_odds & k_tr
    cobertura = len(comunes) / len(k_tr) if k_tr else 0.0

    if not comunes:
        r.fallo("CRUCE VACIO: ningun partido coincide entre fuentes. "
                "Casi con seguridad la canonizacion de nombres de equipo difiere. "
                "Anada las variantes a ALIAS_EQUIPOS en scrapers/betplay.py.")
    elif cobertura < 0.5:
        r.aviso(f"Cobertura baja: solo {len(comunes)}/{len(k_tr)} partidos cruzan "
                f"({cobertura:.0%}). Revise los nombres huerfanos.")
    else:
        r.ok(f"{len(comunes)}/{len(k_tr)} partidos cruzan ({cobertura:.0%}).")

    huerfanos = k_tr - k_odds
    if huerfanos:
        nombres = {f"{t['equipo_local']} vs {t['equipo_visitante']}"
                   for t in trends if t["match_key"] in huerfanos}
        print("           Tendencias sin cuota asociada:")
        for n in sorted(nombres)[:10]:
            print(f"             - {n}")

    # El cruce fino (mercado, seleccion, linea) es el que decide si hay apuestas.
    k1 = {(o["match_key"], o["mercado"], o["seleccion"], o.get("linea")) for o in odds}
    k2 = {(t["match_key"], t["mercado"], t["seleccion"], t.get("linea")) for t in trends}
    finos = k1 & k2
    if not finos:
        r.fallo("Ninguna tendencia casa con una seleccion cotizada "
                "(mercado/seleccion/linea). Revise el mapeo de mercados y lineas.")
    else:
        r.ok(f"{len(finos)} selecciones evaluables (mercado+seleccion+linea).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validacion previa al despliegue.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--offline", action="store_true",
                    help="Omite red y robots; usa fixtures para el contrato.")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    print("=" * 72)
    print("PREFLIGHT — validacion previa a la captura en vivo")
    print("=" * 72)
    cfg = load_config(a.config)
    r = Reporte()

    check_config(cfg, r)
    if not a.offline:
        check_red(cfg, r)
        check_robots(cfg, r)
    else:
        print("\n2-3. Red y robots.txt: OMITIDOS (--offline)")
    odds, trends = check_contrato(cfg, r, a.offline)
    check_cruce(odds, trends, r)

    print("\n" + "=" * 72)
    if r.fallos:
        print(f"RESULTADO: NO-GO — {len(r.fallos)} bloqueo(s), {len(r.avisos)} aviso(s)")
        for f in r.fallos:
            print(f"  · {f}")
        print("\nNo active el cron hasta resolver los bloqueos.")
        return 1
    print(f"RESULTADO: GO — 0 bloqueos, {len(r.avisos)} aviso(s)")
    print("\nPuede activar la captura continua:")
    print("  */5 * * * * cd $(pwd) && ./scripts/capture.py >> logs/capture.log 2>&1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
