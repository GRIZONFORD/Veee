"""
scrapers/betplay.py — Extraccion de cuotas de BetPlay (LaLiga).

Contrato de salida (una fila por SELECCION, no por partido):
    {match_key, liga, fecha_evento, equipo_local, equipo_visitante,
     mercado, seleccion, linea, cuota, ts_captura, horas_al_inicio}

Estrategia de extraccion
------------------------
BetPlay (plataforma Kambi) sirve el arbol de eventos por una API JSON interna;
cuando esta disponible es preferible al parseo del DOM porque es estable,
tipada y mucho mas ligera para el servidor. El adaptador soporta ambas rutas:

  1. `mode: "api"`  -> consume el endpoint JSON configurado y lo aplana con un
     mapa de rutas declarativo (`json_paths` en config.yaml).
  2. `mode: "html"` -> parsea el DOM con selectores CSS configurables.

Ambos desembocan en `_normalize`, de modo que el resto del pipeline es agnostico
a la fuente. Los selectores/rutas DEBEN calibrarse contra el sitio en vivo antes
de la recoleccion definitiva; `python scripts/run_daily.py --dry-run` valida el
contrato sin tocar la red usando `data/fixtures/betplay_sample.json`.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from .base import BaseScraper

log = logging.getLogger(__name__)

# Normalizacion de nomenclatura: el emparejamiento BetPlay <-> Linemate falla si
# los nombres de equipo no se canonizan. Ampliar segun se detecten variantes.
ALIAS_EQUIPOS: dict[str, str] = {
    "atletico de madrid": "Atletico Madrid", "atlético madrid": "Atletico Madrid",
    "athletic de bilbao": "Athletic Club", "athletic bilbao": "Athletic Club",
    "fc barcelona": "Barcelona", "barcelona fc": "Barcelona",
    "real madrid cf": "Real Madrid", "real sociedad de futbol": "Real Sociedad",
    "rcd espanyol": "Espanyol", "rc celta": "Celta Vigo", "celta de vigo": "Celta Vigo",
    "deportivo alaves": "Alaves", "deportivo alavés": "Alaves",
    "rayo vallecano de madrid": "Rayo Vallecano", "real betis balompie": "Real Betis",
    "villarreal cf": "Villarreal", "valencia cf": "Valencia", "sevilla fc": "Sevilla",
    "getafe cf": "Getafe", "girona fc": "Girona", "ca osasuna": "Osasuna",
    "ud las palmas": "Las Palmas", "rcd mallorca": "Mallorca",
}

MERCADOS = {"1X2", "corners_ou", "cards_ou", "ah", "prop_player"}


def canon_equipo(nombre: str) -> str:
    """Canoniza el nombre de un equipo para permitir el emparejamiento de fuentes."""
    limpio = " ".join(nombre.strip().split())
    return ALIAS_EQUIPOS.get(limpio.lower(), limpio)


def match_key(liga: str, fecha_evento: str, local: str, visitante: str) -> str:
    """Clave estable e independiente de la fuente (hash del cuarteto canonico)."""
    dia = fecha_evento[:10]
    raw = f"{liga}|{dia}|{canon_equipo(local)}|{canon_equipo(visitante)}".lower()
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _dig(obj: Any, path: str) -> Any:
    """Accede a rutas anidadas tipo 'events.0.event.name' en estructuras JSON."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = cur[int(part)] if part.isdigit() else cur.get(part)
    return cur


class BetPlayScraper(BaseScraper):
    """Adaptador de cuotas de BetPlay."""

    fuente = "BetPlay"

    def fetch(self, fecha: str) -> list[dict[str, Any]]:
        """Devuelve todas las selecciones cotizadas para la fecha `fecha`."""
        if self.fixture is not None:
            raw = self._load_fixture()
            log.info("BetPlay: modo fixture (%s).", self.fixture)
        elif self.cfg.get("mode", "api") == "api":
            raw = self._fetch_api(fecha)
        else:
            raw = self._fetch_html(fecha)
        rows = self._normalize(raw)
        log.info("BetPlay: %d selecciones normalizadas para %s.", len(rows), fecha)
        return rows

    # ------------------------------------------------------------- captura --
    def _fetch_api(self, fecha: str) -> list[dict[str, Any]]:
        url = self.cfg["api_url"]
        params = dict(self.cfg.get("api_params", {}))
        params.setdefault("date", fecha)
        payload = self.session.get(url, params=params).json()
        eventos = _dig(payload, self.cfg.get("json_paths", {}).get("events", "events")) or []
        out: list[dict[str, Any]] = []
        jp = self.cfg.get("json_paths", {})
        for ev in eventos:
            out.append({
                "equipo_local": _dig(ev, jp.get("home", "event.homeName")),
                "equipo_visitante": _dig(ev, jp.get("away", "event.awayName")),
                "fecha_evento": _dig(ev, jp.get("start", "event.start")),
                "offers": _dig(ev, jp.get("offers", "betOffers")) or [],
            })
        return out

    def _fetch_html(self, fecha: str) -> list[dict[str, Any]]:
        sel = self.cfg["selectors"]
        html = self.session.get(self.cfg["html_url"], params={"date": fecha}).text
        soup = self.session.soup(html)
        out: list[dict[str, Any]] = []
        for card in soup.select(sel["event_card"]):
            equipos = [e.get_text(strip=True) for e in card.select(sel["team"])]
            if len(equipos) < 2:
                continue
            offers = []
            for row in card.select(sel["offer_row"]):
                etiquetas = [x.get_text(strip=True) for x in row.select(sel["outcome_label"])]
                cuotas = [x.get_text(strip=True) for x in row.select(sel["outcome_odds"])]
                crit = row.select_one(sel["criterion"])
                offers.append({
                    "criterion": crit.get_text(strip=True) if crit else "",
                    "outcomes": [{"label": l, "odds": c} for l, c in zip(etiquetas, cuotas)],
                })
            out.append({
                "equipo_local": equipos[0], "equipo_visitante": equipos[1],
                "fecha_evento": (card.select_one(sel["start_time"]) or {}).get("datetime", fecha)
                if hasattr(card.select_one(sel["start_time"]), "get") else fecha,
                "offers": offers,
            })
        return out

    # --------------------------------------------------------- normalizacion --
    def _normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aplana a una fila por seleccion y valida rangos de cuota."""
        ts = self.now_iso()
        ahora = datetime.now(timezone.utc)
        liga = self.cfg.get("liga", "LaLiga")
        filas: list[dict[str, Any]] = []
        for ev in raw:
            local = canon_equipo(str(ev["equipo_local"]))
            visitante = canon_equipo(str(ev["equipo_visitante"]))
            f_evento = str(ev["fecha_evento"])
            try:
                dt = datetime.fromisoformat(f_evento.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                horas = (dt - ahora).total_seconds() / 3600.0
            except ValueError:
                horas = None
            mk = match_key(liga, f_evento, local, visitante)
            for offer in ev.get("offers", []):
                mercado, linea = self._map_mercado(str(offer.get("criterion", "")))
                if mercado is None:
                    continue
                for oc in offer.get("outcomes", []):
                    cuota = self._parse_cuota(oc.get("odds"))
                    if cuota is None:
                        continue
                    filas.append({
                        "match_key": mk, "liga": liga, "fecha_evento": f_evento,
                        "equipo_local": local, "equipo_visitante": visitante,
                        "mercado": mercado,
                        "seleccion": self._map_seleccion(str(oc.get("label", "")), mercado),
                        "linea": oc.get("line", linea),
                        "cuota": cuota, "ts_captura": ts, "horas_al_inicio": horas,
                        "casa": self.fuente,
                    })
        return filas

    @staticmethod
    def _parse_cuota(v: Any) -> float | None:
        """Acepta 2.10, '2.10', '2,10' o formato Kambi en milesimas (2100)."""
        if v is None:
            return None
        try:
            x = float(str(v).replace(",", "."))
        except ValueError:
            return None
        if x > 100:                       # heuristica Kambi: odds * 1000
            x /= 1000.0
        return x if 1.01 <= x <= 1001.0 else None

    @staticmethod
    def _map_mercado(criterion: str) -> tuple[str | None, float | None]:
        """Traduce la etiqueta comercial del mercado a la taxonomia del estudio."""
        c = criterion.lower()
        linea = None
        for token in c.replace(",", ".").split():
            try:
                linea = float(token.strip("()+"))
                break
            except ValueError:
                continue
        if "corner" in c or "esquina" in c:
            return "corners_ou", linea
        if "tarjeta" in c or "card" in c or "amonesta" in c:
            return "cards_ou", linea
        if "handicap" in c or "hándicap" in c or "linea" in c or "spread" in c:
            return "ah", linea
        if "1x2" in c or "resultado" in c or "match odds" in c or "ganador" in c:
            return "1X2", None
        if "jugador" in c or "player" in c or "anotador" in c or "disparo" in c:
            return "prop_player", linea
        return None, None

    @staticmethod
    def _map_seleccion(label: str, mercado: str) -> str:
        l = label.lower().strip()
        if mercado == "1X2":
            if l in {"1", "local", "home"}:
                return "home"
            if l in {"x", "empate", "draw"}:
                return "draw"
            if l in {"2", "visitante", "away"}:
                return "away"
        if "mas de" in l or "más de" in l or l.startswith("over") or l == "o":
            return "over"
        if "menos de" in l or l.startswith("under") or l == "u":
            return "under"
        return label.strip()
