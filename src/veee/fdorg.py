r"""
fdorg.py — Conector de football-data.org (API v4).

QUE APORTA Y QUE NO
-------------------
football-data.org **no sirve cuotas** en el plan gratuito (el campo `odds` de
cada partido devuelve un mensaje pidiendo activar un paquete de pago). No
sustituye a Football-Data.co.uk, que es de donde salen los precios del estudio.

Lo que aporta es igualmente valioso, y resuelve tres problemas concretos:

1. **Identificadores estables de equipo.** Cada club tiene un `id` numerico y un
   `tla` de tres letras invariables entre temporadas. Es el ancla que faltaba
   para el emparejamiento entre fuentes: Football-Data.co.uk escribe
   "Ath Bilbao", Polymarket "Athletic Club" y una casa "Athletic de Bilbao". Un
   emparejamiento erroneo no produce un error, produce una discrepancia de precio
   espuria enorme, y era el fallo silencioso mas peligroso del diseno.
2. **Calendario con hora exacta en UTC.** Sin la hora del pitido inicial la
   escalera de captura no puede anclarse y no hay linea de cierre, luego no hay
   CLV.
3. **Resultados para liquidar**, incluido el marcador al descanso.

No confundir con **football-data.co.uk** (sin punto org), que es el archivo
gratuito de CSV con las cuotas historicas. Son servicios distintos.

SEGURIDAD DE LA CREDENCIAL
--------------------------
El token se lee **solo** de la variable de entorno `FOOTBALL_DATA_ORG_TOKEN` y
nunca se escribe en disco ni aparece completo en los registros: `_mascara()` deja
unicamente los cuatro ultimos caracteres. Un token filtrado en un repositorio
publico queda expuesto de forma permanente en el historial de git.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger(__name__)

BASE = "https://api.football-data.org/v4"
ENV_TOKEN = "FOOTBALL_DATA_ORG_TOKEN"

# Codigos de competicion. Los del plan gratuito son un subconjunto que conviene
# confirmar con `verificar()`: el plan puede cambiar sin aviso.
COMPETICIONES: dict[str, str] = {
    "PD": "LaLiga (Primera Division)", "PL": "Premier League",
    "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1",
    "DED": "Eredivisie", "PPL": "Primeira Liga", "ELC": "Championship",
    "CL": "Champions League", "EC": "Eurocopa", "WC": "Mundial",
    "BSA": "Brasileirao",
}

# Correspondencia con los codigos de division de Football-Data.co.uk, para poder
# cruzar ambas fuentes por liga.
A_DIV_ARCHIVO: dict[str, str] = {
    "PD": "SP1", "PL": "E0", "BL1": "D1", "SA": "I1",
    "FL1": "F1", "DED": "N1", "PPL": "P1", "ELC": "E1",
}

RESULTADO_FT = {"HOME_TEAM": "H", "AWAY_TEAM": "A", "DRAW": "D"}


class FootballDataOrgError(RuntimeError):
    """Error de la API con diagnostico accionable."""


def _mascara(token: str) -> str:
    return f"...{token[-4:]}" if len(token) > 4 else "****"


@dataclass
class LimiteTasa:
    """Control del limite del plan gratuito (10 peticiones por minuto).

    Excederlo devuelve 429 y, si se insiste, puede bloquear la clave. El
    espaciado uniforme es preferible a agotar la rafaga y esperar: mantiene el
    proceso predecible y no depende de que el servidor informe bien del reinicio.
    """
    por_minuto: int = 10
    _ultima: float = 0.0

    @property
    def intervalo(self) -> float:
        return 60.0 / max(self.por_minuto, 1)

    def esperar(self) -> None:
        falta = self.intervalo - (time.monotonic() - self._ultima)
        if falta > 0:
            log.debug("Limite de tasa: esperando %.1fs", falta)
            time.sleep(falta)
        self._ultima = time.monotonic()


class FootballDataOrg:
    """Cliente de la API v4. Mantiene estado: token, limite de tasa y contadores."""

    def __init__(self, token: str | None = None, por_minuto: int = 10,
                 timeout: int = 20, max_reintentos: int = 3) -> None:
        self.token = token or os.getenv(ENV_TOKEN, "")
        if not self.token:
            raise FootballDataOrgError(
                f"Falta el token. Exportelo antes de ejecutar:\n"
                f"    export {ENV_TOKEN}='su_clave'\n"
                "Nunca lo escriba en un fichero del repositorio."
            )
        self.limite = LimiteTasa(por_minuto)
        self.timeout = timeout
        self.max_reintentos = max_reintentos
        self.n_peticiones = 0
        self.restantes_minuto: int | None = None
        import requests
        self.sesion = requests.Session()
        self.sesion.headers.update({
            "X-Auth-Token": self.token,
            "User-Agent": "VeeeAcademicResearch/1.0 (estudio universitario de economia)",
        })
        log.info("Cliente football-data.org listo (token %s).", _mascara(self.token))

    # ------------------------------------------------------------- peticion --
    def get(self, ruta: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET con control de tasa y diagnostico por codigo de estado."""
        import requests
        url = f"{BASE}/{ruta.lstrip('/')}"
        for intento in range(self.max_reintentos):
            self.limite.esperar()
            try:
                r = self.sesion.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if intento == self.max_reintentos - 1:
                    raise FootballDataOrgError(f"Fallo de red en {ruta}: {exc}") from exc
                time.sleep(2.0 ** intento)
                continue
            self.n_peticiones += 1
            if (dis := r.headers.get("X-Requests-Available-Minute")) is not None:
                try:
                    self.restantes_minuto = int(dis)
                except ValueError:
                    pass

            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                raise FootballDataOrgError(
                    f"Peticion invalida en {ruta}: {r.text[:200]}")
            if r.status_code == 401:
                raise FootballDataOrgError(
                    f"Token rechazado ({_mascara(self.token)}). Revise que la clave "
                    "sea correcta y este activada en su panel de usuario.")
            if r.status_code == 403:
                raise FootballDataOrgError(
                    f"Recurso fuera de su plan: {ruta}. El plan gratuito limita "
                    "competiciones y temporadas. Ejecute 'verificar' para ver que "
                    "cubre exactamente su clave.")
            if r.status_code == 404:
                raise FootballDataOrgError(f"No existe: {ruta}")
            if r.status_code == 429:
                espera = float(r.headers.get("X-RequestCounter-Reset", 60)) + 1
                log.warning("Limite de tasa alcanzado; esperando %.0fs.", espera)
                time.sleep(min(espera, 90))
                continue
            if 500 <= r.status_code < 600:
                time.sleep(2.0 ** intento)
                continue
            raise FootballDataOrgError(f"HTTP {r.status_code} en {ruta}: {r.text[:200]}")
        raise FootballDataOrgError(f"Agotados los reintentos en {ruta}")

    # ---------------------------------------------------------- verificacion --
    def verificar(self, competiciones: Iterable[str] = ("PD",),
                  temporadas: Iterable[int] = (2024,)) -> dict[str, Any]:
        """Comprueba el token y **mide** que cubre el plan, en vez de suponerlo.

        Prueba cada combinacion competicion-temporada y registra si responde. Es
        el equivalente al informe de cobertura del archivo: convierte los limites
        del plan en un hecho observado.
        """
        informe: dict[str, Any] = {"token": _mascara(self.token), "acceso": {},
                                   "competiciones_visibles": [], "errores": []}
        try:
            comps = self.get("competitions")
            informe["competiciones_visibles"] = [
                {"code": c.get("code"), "name": c.get("name")}
                for c in comps.get("competitions", []) if c.get("code")
            ]
        except FootballDataOrgError as exc:
            informe["errores"].append(f"listado de competiciones: {exc}")

        for comp in competiciones:
            for temp in temporadas:
                clave = f"{comp}:{temp}"
                try:
                    p = self.get(f"competitions/{comp}/matches",
                                 {"season": temp})
                    rs = p.get("resultSet", {})
                    informe["acceso"][clave] = {
                        "ok": True, "partidos": rs.get("count", len(p.get("matches", []))),
                        "jugados": rs.get("played"),
                        "primero": rs.get("first"), "ultimo": rs.get("last"),
                        "trae_cuotas": _trae_cuotas(p),
                    }
                except FootballDataOrgError as exc:
                    informe["acceso"][clave] = {"ok": False, "motivo": str(exc)}
        informe["peticiones_realizadas"] = self.n_peticiones
        informe["restantes_minuto"] = self.restantes_minuto
        return informe

    # -------------------------------------------------------------- partidos --
    def partidos(self, competicion: str, temporada: int) -> dict[str, Any]:
        """Payload crudo de /competitions/{code}/matches?season=YYYY."""
        return self.get(f"competitions/{competicion}/matches", {"season": temporada})


def _trae_cuotas(payload: dict[str, Any]) -> bool:
    """¿El plan incluye cuotas? En el gratuito, `odds` trae solo un aviso."""
    for m in payload.get("matches", [])[:5]:
        odds = m.get("odds") or {}
        if isinstance(odds, dict) and any(
                k in odds for k in ("homeWin", "draw", "awayWin")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Normalizacion al mismo contrato que archive.py                               #
# --------------------------------------------------------------------------- #
def normalizar_partidos(payload: dict[str, Any]) -> pd.DataFrame:
    """Aplana el payload al contrato de `archive.normalizar_partidos`.

    Conserva ademas `home_id`/`away_id` y los `tla`, que son los identificadores
    estables que permiten anclar el emparejamiento entre fuentes.

    Solo se marcan como jugados los partidos con estado FINISHED: un aplazado o
    suspendido con marcador nulo se colaria como 0-0 si se leyera sin filtrar.
    """
    comp = payload.get("competition", {}) or {}
    code = comp.get("code")
    filas = []
    for m in payload.get("matches", []) or []:
        h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        sc = m.get("score") or {}
        ft, ht = sc.get("fullTime") or {}, sc.get("halfTime") or {}
        finalizado = m.get("status") == "FINISHED"
        fecha = pd.to_datetime(m.get("utcDate"), errors="coerce", utc=True)
        filas.append({
            "fd_match_id": m.get("id"),
            "div": A_DIV_ARCHIVO.get(code, code),
            "competicion": code,
            "liga": comp.get("name"),
            "temporada": _etiqueta_temporada(payload, fecha),
            "jornada": m.get("matchday"),
            "fecha": fecha,
            "estado": m.get("status"),
            "equipo_local": h.get("name"), "equipo_visitante": a.get("name"),
            "local_corto": h.get("shortName"), "visitante_corto": a.get("shortName"),
            "local_tla": h.get("tla"), "visitante_tla": a.get("tla"),
            "home_id": h.get("id"), "away_id": a.get("id"),
            "goles_local": ft.get("home") if finalizado else None,
            "goles_visitante": ft.get("away") if finalizado else None,
            "goles_local_ht": ht.get("home") if finalizado else None,
            "goles_visitante_ht": ht.get("away") if finalizado else None,
            "resultado_ft": RESULTADO_FT.get(sc.get("winner")) if finalizado else None,
        })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["goles_totales"] = df["goles_local"] + df["goles_visitante"]
    df["margen_goles"] = df["goles_local"] - df["goles_visitante"]
    return df


def _etiqueta_temporada(payload: dict[str, Any], fecha) -> str | None:
    """Etiqueta '2024-25' a partir de los filtros o de la fecha del partido."""
    temp = (payload.get("filters") or {}).get("season")
    if temp:
        try:
            a = int(str(temp)[:4])
            return f"{a}-{(a + 1) % 100:02d}"
        except ValueError:
            pass
    if pd.notna(fecha):
        a = fecha.year if fecha.month >= 7 else fecha.year - 1
        return f"{a}-{(a + 1) % 100:02d}"
    return None


def tabla_equipos(df: pd.DataFrame) -> pd.DataFrame:
    """Catalogo de equipos con sus identificadores estables.

    Es el insumo del emparejamiento entre fuentes: un `id` numerico que no cambia
    entre temporadas es infinitamente mas fiable que comparar cadenas de texto.
    """
    local = df[["home_id", "equipo_local", "local_corto", "local_tla"]].rename(
        columns={"home_id": "team_id", "equipo_local": "nombre",
                 "local_corto": "nombre_corto", "local_tla": "tla"})
    visita = df[["away_id", "equipo_visitante", "visitante_corto", "visitante_tla"]].rename(
        columns={"away_id": "team_id", "equipo_visitante": "nombre",
                 "visitante_corto": "nombre_corto", "visitante_tla": "tla"})
    return (pd.concat([local, visita], ignore_index=True)
              .dropna(subset=["team_id"]).drop_duplicates("team_id")
              .sort_values("nombre").reset_index(drop=True))
