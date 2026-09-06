"""
scrapers/base.py — Cliente HTTP responsable y contrato comun de los extractores.

Consideraciones eticas y legales (relevantes para la seccion de Metodologia)
---------------------------------------------------------------------------
La extraccion automatizada se limita a informacion de precios publicamente
accesible, con fines exclusivamente academicos y sin reproduccion comercial. El
cliente: (i) consulta y respeta `robots.txt`; (ii) impone un retardo minimo entre
peticiones (*rate limiting*) para no degradar el servicio; (iii) identifica al
agente con un User-Agent honesto de contacto academico; (iv) archiva la respuesta
cruda para permitir replicacion sin re-scraping. Se recomienda documentar en el
paper la fecha de consulta de los Terminos de Servicio de cada plataforma.

Advertencia de implementacion
-----------------------------
Los selectores CSS y los *endpoints* concretos de BetPlay y Linemate NO son
estables ni documentados publicamente: cambian sin previo aviso. Por ello el
diseno es **dirigido por configuracion** (`config/config.yaml`): el codigo fija
el contrato de datos y la logica economica; los selectores se calibran contra el
sitio en vivo y se validan con `--dry-run` sobre los *fixtures* de `data/fixtures/`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

USER_AGENT = (
    "VeeeAcademicResearchBot/1.0 (+proyecto universitario de economia; "
    "estudio de eficiencia de mercados; contacto: jhguzman@unal.edu.co)"
)


class PoliteSession:
    """Sesion HTTP con robots.txt, rate limiting, reintentos y cache en disco."""

    def __init__(self, rate_limit_s: float = 2.0, timeout: int = 20,
                 max_retries: int = 4, respect_robots: bool = True,
                 cache_dir: str | Path = "data/raw") -> None:
        self.rate_limit_s = rate_limit_s
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0
        self._robots: dict[str, RobotFileParser] = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "es-CO,es;q=0.9",
        })

    # ---------------------------------------------------------------- robots --
    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception as exc:  # sin robots.txt accesible -> se asume permitido
                log.warning("robots.txt inaccesible en %s (%s); se continua con cautela.", base, exc)
                rp = RobotFileParser()
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------ throttling --
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self.rate_limit_s - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.4))  # jitter anti-sincronizacion
        self._last_call = time.monotonic()

    # ------------------------------------------------------------------ get --
    def get(self, url: str, params: dict[str, Any] | None = None,
            cache: bool = True) -> requests.Response:
        """GET con backoff exponencial. Archiva la respuesta cruda si `cache`."""
        if not self._allowed(url):
            raise PermissionError(f"robots.txt prohibe el acceso automatizado a {url}")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                if cache:
                    self._archive(url, params, r.text)
                return r
            except Exception as exc:            # noqa: BLE001 - reintento generico
                last_exc = exc
                backoff = 2.0 ** attempt
                log.warning("Fallo GET %s (intento %d/%d): %s. Reintento en %.0fs",
                            url, attempt + 1, self.max_retries, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"GET fallido tras {self.max_retries} intentos: {url}") from last_exc

    def _archive(self, url: str, params: dict[str, Any] | None, text: str) -> Path:
        key = hashlib.sha256(f"{url}{params}".encode()).hexdigest()[:16]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.cache_dir / f"{stamp}_{key}.html"
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml")


class BaseScraper(ABC):
    """Contrato comun. Toda subclase entrega registros ya normalizados."""

    fuente: str = "generic"

    def __init__(self, cfg: dict[str, Any], session: PoliteSession | None = None,
                 fixture: str | Path | None = None) -> None:
        self.cfg = cfg
        self.session = session or PoliteSession(
            rate_limit_s=cfg.get("rate_limit_s", 2.0),
            respect_robots=cfg.get("respect_robots", True),
        )
        self.fixture = Path(fixture) if fixture else None

    def _load_fixture(self) -> Any:
        """Modo offline: reproduce una captura archivada (tests y `--dry-run`)."""
        if self.fixture is None:
            raise FileNotFoundError("No hay fixture configurado.")
        return json.loads(self.fixture.read_text(encoding="utf-8"))

    @abstractmethod
    def fetch(self, fecha: str) -> list[dict[str, Any]]:
        """Devuelve registros normalizados para la fecha dada (ISO YYYY-MM-DD)."""

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
