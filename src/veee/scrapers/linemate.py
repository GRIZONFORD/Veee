"""
scrapers/linemate.py — Extraccion de tendencias/metricas de Linemate.

Contrato de salida (una fila por TENDENCIA):
    {match_key, mercado, seleccion, linea, hits, n_muestra, tasa_bruta,
     media_reciente, varianza_reciente, trend_score, ts_captura}

Naturaleza de la senal
----------------------
Linemate publica tendencias del tipo "el equipo X ha superado 9.5 corners en 8 de
sus ultimos 10 partidos". Economicamente esto es un estadistico de frecuencia
sobre una ventana movil corta: informativo pero **ruidoso** y potencialmente ya
incorporado en el precio. El extractor conserva `hits` y `n_muestra` por separado
(no solo la tasa) precisamente para poder aplicar despues el encogimiento
empirico-bayesiano de `oddsmath.shrink_to_prior`, que es lo que impide confundir
una racha con una ventaja.

El `trend_score` se define como el estadistico z de la tasa observada frente a la
probabilidad neutralizada del mercado:

    trend_score = (tasa_bruta - p_mercado) / sqrt(p_mercado (1-p_mercado)/n)

de modo que sea comparable entre mercados y directamente interpretable como
"desviaciones estandar de discrepancia frente al precio".
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

from .base import BaseScraper
from .betplay import canon_equipo, match_key

log = logging.getLogger(__name__)

# "8 of last 10" / "8 de los ultimos 10" / "8/10"
_RE_HITS = re.compile(r"(\d+)\s*(?:of|de(?:\s+los)?|/)\s*(?:last|sus\s+ultimos|ultimos\s*)?\s*(\d+)",
                      re.IGNORECASE)
_RE_LINEA = re.compile(r"(\d+(?:[.,]\d+)?)")


def parse_trend_text(texto: str) -> dict[str, Any] | None:
    """Extrae (hits, n_muestra, linea, sentido) de una tendencia en lenguaje natural."""
    m = _RE_HITS.search(texto)
    if not m:
        return None
    hits, n = int(m.group(1)), int(m.group(2))
    if n <= 0 or hits > n:
        return None
    t = texto.lower()
    es_under = "under" in t or "menos de" in t or "por debajo" in t
    sentido = "under" if es_under else "over"
    linea = None
    mm = _RE_LINEA.search(texto.replace(m.group(0), ""))
    if mm:
        try:
            linea = float(mm.group(1).replace(",", "."))
        except ValueError:
            linea = None
    return {"hits": hits, "n_muestra": n, "linea": linea, "seleccion": sentido,
            "tasa_bruta": hits / n}


def trend_z(tasa_bruta: float, n: int, p_mercado: float) -> float:
    r"""Estadistico z de la tendencia frente al precio neutralizado del mercado.

    .. math::
        \text{TrendScore} = \frac{\hat{p}_{trend} - p_{mercado}}
                                 {\sqrt{p_{mercado}(1-p_{mercado})/n}}
    """
    p = min(max(p_mercado, 1e-6), 1 - 1e-6)
    se = math.sqrt(p * (1 - p) / max(n, 1))
    return (tasa_bruta - p) / se if se > 0 else 0.0


class LinemateScraper(BaseScraper):
    """Adaptador de tendencias de Linemate."""

    fuente = "Linemate"

    def fetch(self, fecha: str) -> list[dict[str, Any]]:
        if self.fixture is not None:
            raw = self._load_fixture()
            log.info("Linemate: modo fixture (%s).", self.fixture)
        elif self.cfg.get("mode", "api") == "api":
            url = self.cfg["api_url"]
            params = dict(self.cfg.get("api_params", {}))
            params.setdefault("date", fecha)
            raw = self.session.get(url, params=params).json()
            raw = raw.get(self.cfg.get("json_root", "trends"), raw) if isinstance(raw, dict) else raw
        else:
            raw = self._fetch_html(fecha)
        rows = self._normalize(raw)
        log.info("Linemate: %d tendencias normalizadas para %s.", len(rows), fecha)
        return rows

    def _fetch_html(self, fecha: str) -> list[dict[str, Any]]:
        sel = self.cfg["selectors"]
        html = self.session.get(self.cfg["html_url"], params={"date": fecha}).text
        soup = self.session.soup(html)
        out: list[dict[str, Any]] = []
        for card in soup.select(sel["trend_card"]):
            equipos = [e.get_text(strip=True) for e in card.select(sel["team"])]
            if len(equipos) < 2:
                continue
            for item in card.select(sel["trend_item"]):
                out.append({
                    "equipo_local": equipos[0], "equipo_visitante": equipos[1],
                    "fecha_evento": fecha,
                    "mercado_raw": (card.select_one(sel["market"]).get_text(strip=True)
                                    if card.select_one(sel["market"]) else ""),
                    "texto": item.get_text(" ", strip=True),
                })
        return out

    def _normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ts = self.now_iso()
        liga = self.cfg.get("liga", "LaLiga")
        filas: list[dict[str, Any]] = []
        for r in raw:
            parsed = parse_trend_text(str(r.get("texto", ""))) if "texto" in r else None
            if parsed is None:
                # Modo API: los campos ya vienen estructurados.
                if r.get("hits") is None or not r.get("n_muestra"):
                    continue
                parsed = {"hits": int(r["hits"]), "n_muestra": int(r["n_muestra"]),
                          "linea": r.get("linea"), "seleccion": r.get("seleccion", "over"),
                          "tasa_bruta": float(r["hits"]) / float(r["n_muestra"])}
            mercado = self._map_mercado(str(r.get("mercado_raw", r.get("mercado", ""))))
            if mercado is None:
                continue
            local = canon_equipo(str(r["equipo_local"]))
            visitante = canon_equipo(str(r["equipo_visitante"]))
            filas.append({
                "match_key": match_key(liga, str(r["fecha_evento"]), local, visitante),
                "equipo_local": local, "equipo_visitante": visitante,
                "fecha_evento": str(r["fecha_evento"]),
                "mercado": mercado, "seleccion": parsed["seleccion"],
                "linea": parsed["linea"] if parsed["linea"] is not None else r.get("linea"),
                "hits": parsed["hits"], "n_muestra": parsed["n_muestra"],
                "tasa_bruta": parsed["tasa_bruta"],
                "media_reciente": r.get("media_reciente"),
                "varianza_reciente": r.get("varianza_reciente"),
                "ts_captura": ts, "fuente": self.fuente,
            })
        return filas

    @staticmethod
    def _map_mercado(texto: str) -> str | None:
        t = texto.lower()
        if "corner" in t or "esquina" in t:
            return "corners_ou"
        if "card" in t or "tarjeta" in t:
            return "cards_ou"
        if "spread" in t or "handicap" in t or "hándicap" in t:
            return "ah"
        if "moneyline" in t or "1x2" in t or "resultado" in t:
            return "1X2"
        if "player" in t or "jugador" in t or "shot" in t or "disparo" in t:
            return "prop_player"
        return None
