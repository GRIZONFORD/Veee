"""
capture.py — Programador de captura continua: escalera de ventanas y linea de cierre.

Por que no basta con una captura diaria
---------------------------------------
El CLV es el contraste de mayor potencia del estudio (ver docs/METODOLOGIA.md) y
exige la **linea de cierre**: el ultimo precio vigente antes del pitido inicial.
Una unica captura diaria jamas la obtiene. Este modulo implementa una escalera de
ventanas ancladas al tiempo restante hasta el inicio del partido, de modo que:

    T-72h  T-48h  T-24h  T-12h  T-6h  T-3h  T-1h  T-30m  T-10m  T-3m
      |      |      |      |     |     |     |      |      |      |
      +------+------+---- apertura --> convergencia ------>+   CIERRE

Cada ventana se captura **como maximo una vez** por partido (registro en
`capture_log`), lo que hace el proceso idempotente y seguro frente a
reejecuciones de cron.

Separacion entre capturar y apostar
-----------------------------------
Se capturan precios en toda la escalera, pero las apuestas se registran solo
dentro de la **ventana de apuesta** pre-registrada (por defecto T-24h a T-6h).
Fijar el momento de colocacion es un requisito del protocolo: si se apostara en
cualquier instante, el CLV mediria en parte la eleccion del momento y no la
calidad de la senal, y el estudio dejaria de ser comparable entre partidos.

Deteccion de rupturas silenciosas
---------------------------------
Un cambio en el DOM de la casa no produce un error: produce **cero filas**. Eso
introduciria datos faltantes no aleatorios que sesgarian la muestra sin dejar
rastro. `health_check` compara el volumen capturado contra la mediana movil
reciente y falla de forma ruidosa (codigo de salida distinto de cero) cuando cae
por debajo del umbral, para que cron o el runner lo notifiquen.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from . import database as db
from .oddsmath import clv_odds_ratio, devig

log = logging.getLogger(__name__)

# Escalera de ventanas, en horas hasta el inicio. Densa cerca del pitido porque
# es donde la linea converge y se juega la calidad del CLV.
VENTANAS_H: tuple[float, ...] = (72, 48, 24, 12, 6, 3, 1, 0.5, 1 / 6, 0.05)

# Tolerancia: una ventana se considera "vencida" si el partido ya paso ese hito.
# Con cron cada 5 minutos ninguna ventana se pierde por desfase del planificador.
TOLERANCIA_H = 0.02

SCHEMA_CAPTURA = """
CREATE TABLE IF NOT EXISTS capture_log (
    match_id     TEXT NOT NULL,
    ventana_h    REAL NOT NULL,
    fuente       TEXT NOT NULL,
    ts_captura   TEXT NOT NULL,
    n_filas      INTEGER NOT NULL,
    run_id       TEXT NOT NULL,
    PRIMARY KEY (match_id, ventana_h, fuente)
);
CREATE TABLE IF NOT EXISTS health_log (
    ts           TEXT NOT NULL,
    fuente       TEXT NOT NULL,
    n_filas      INTEGER NOT NULL,
    mediana_ref  REAL,
    estado       TEXT NOT NULL,
    detalle      TEXT
);
"""


@dataclass
class CaptureParams:
    """Parametros operativos de la captura (bloque `captura` de config.yaml)."""
    ventanas_h: tuple[float, ...] = VENTANAS_H
    bet_window: tuple[float, float] = (24.0, 6.0)   # (desde, hasta) horas al inicio
    horizonte_h: float = 96.0                       # cuanto futuro se vigila
    salud_umbral: float = 0.5                       # fraccion de la mediana movil
    salud_min_muestras: int = 5


def init_capture_schema(db_path: str) -> None:
    with db.connect(db_path) as conn:
        conn.executescript(SCHEMA_CAPTURA)


# --------------------------------------------------------------------------- #
# 1. Planificacion: que partidos y que ventanas tocan ahora                    #
# --------------------------------------------------------------------------- #
def horas_al_inicio(fecha_evento: str, ahora: datetime | None = None) -> float | None:
    """Horas restantes hasta el pitido inicial (negativo si ya comenzo)."""
    ahora = ahora or datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(fecha_evento).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - ahora).total_seconds() / 3600.0


def ventanas_alcanzadas(h_restantes: float, ventanas: Iterable[float],
                        ya_capturadas: set[float]) -> list[float]:
    """Ventanas ya alcanzadas por el reloj y todavia sin capturar."""
    if h_restantes < -TOLERANCIA_H:
        return []
    return sorted(v for v in ventanas
                  if h_restantes <= v + TOLERANCIA_H and v not in ya_capturadas)


def ventana_debida(h_restantes: float, ventanas: Iterable[float],
                   ya_capturadas: set[float]) -> float | None:
    """Ventana pendiente mas ajustada al momento actual, si la hay.

    Se elige la **mas pequena ya alcanzada**: si cron estuvo caido, se recupera el
    hito mas reciente en vez de registrar una cascada de capturas con el mismo
    precio. Las ventanas mayores no capturadas quedan PERDIDAS, no pendientes
    (ver `ventanas_perdidas`): etiquetar a T-3h un precio como si fuera de T-24h
    corromperia el analisis de convergencia de la linea.
    """
    alcanzadas = ventanas_alcanzadas(h_restantes, ventanas, ya_capturadas)
    return alcanzadas[0] if alcanzadas else None


def ventanas_perdidas(h_restantes: float, ventanas: Iterable[float],
                      ya_capturadas: set[float]) -> list[float]:
    """Ventanas que el reloj ya rebaso y que no se capturaran nunca.

    Se registran explicitamente (con `n_filas = -1`) para dejar constancia del
    hueco: los datos faltantes deben ser visibles en la muestra, no invisibles.
    """
    return ventanas_alcanzadas(h_restantes, ventanas, ya_capturadas)[1:]


def en_ventana_de_apuesta(h_restantes: float, par: CaptureParams) -> bool:
    """True si el instante actual cae en la ventana de colocacion pre-registrada."""
    desde, hasta = par.bet_window
    return hasta <= h_restantes <= desde


def partidos_vigilados(db_path: str, par: CaptureParams,
                       ahora: datetime | None = None) -> list[dict[str, Any]]:
    """Partidos programados dentro del horizonte, con su estado de captura."""
    ahora = ahora or datetime.now(timezone.utc)
    limite = (ahora + timedelta(hours=par.horizonte_h)).isoformat()
    with db.connect(db_path) as conn:
        filas = conn.execute(
            "SELECT match_id, fecha_evento, equipo_local, equipo_visitante "
            "FROM matches WHERE estado = 'programado' AND fecha_evento <= ? "
            "ORDER BY fecha_evento", (limite,)
        ).fetchall()
        out = []
        for f in filas:
            h = horas_al_inicio(f["fecha_evento"], ahora)
            if h is None or h < -TOLERANCIA_H:
                continue
            hechas = {r["ventana_h"] for r in conn.execute(
                "SELECT ventana_h FROM capture_log WHERE match_id = ? AND fuente = 'BetPlay'",
                (f["match_id"],)).fetchall()}
            out.append({"match_id": f["match_id"], "fecha_evento": f["fecha_evento"],
                        "partido": f"{f['equipo_local']} vs {f['equipo_visitante']}",
                        "h_restantes": h, "ventanas_hechas": hechas,
                        "ventana_debida": ventana_debida(h, par.ventanas_h, hechas),
                        "ventanas_perdidas": ventanas_perdidas(h, par.ventanas_h, hechas),
                        "en_ventana_apuesta": en_ventana_de_apuesta(h, par)})
        return out


def registrar_captura(db_path: str, match_id: str, ventana_h: float, fuente: str,
                      n_filas: int, run_id: str) -> None:
    with db.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO capture_log "
            "(match_id, ventana_h, fuente, ts_captura, n_filas, run_id) "
            "VALUES (?,?,?,?,?,?)",
            (match_id, ventana_h, fuente,
             datetime.now(timezone.utc).isoformat(timespec="seconds"), n_filas, run_id))


# --------------------------------------------------------------------------- #
# 2. Consolidacion de la linea de cierre y del CLV                             #
# --------------------------------------------------------------------------- #
def finalize_closing_lines(db_path: str, ahora: datetime | None = None,
                           devig_method: str = "shin") -> dict[str, int]:
    """Marca la linea de cierre y calcula el CLV de las apuestas ya iniciadas.

    Para cada partido cuyo pitido inicial ya paso, se toma el ULTIMO snapshot
    anterior al inicio de cada (mercado, linea) y se marca `es_cierre=1`. Con esas
    cuotas se neutraliza el margen y se rellenan `cuota_cierre`, `p_cierre_devig`,
    `clv_odds` y `clv_prob` de las apuestas correspondientes.

    Solo se consideran snapshots ANTERIORES al inicio: usar un precio posterior
    seria contaminacion ex post y destruiria la validez del CLV.
    """
    ahora = ahora or datetime.now(timezone.utc)
    n_cierres = n_clv = 0
    with db.connect(db_path) as conn:
        conn.execute("BEGIN")
        pendientes = conn.execute(
            "SELECT DISTINCT m.match_id, m.fecha_evento FROM matches m "
            "JOIN bets b ON b.match_id = m.match_id "
            "WHERE b.cuota_cierre IS NULL AND m.fecha_evento <= ?",
            (ahora.isoformat(),)).fetchall()

        for m in pendientes:
            mercados = conn.execute(
                "SELECT DISTINCT mercado, IFNULL(linea,-999) AS lin FROM odds_snapshots "
                "WHERE match_id = ? AND ts_captura < ?",
                (m["match_id"], m["fecha_evento"])).fetchall()

            for mk in mercados:
                lin = None if mk["lin"] == -999 else mk["lin"]
                # Ultimo snapshot pre-partido de cada seleccion del mercado.
                filas = conn.execute(
                    "SELECT seleccion, cuota, snapshot_id, ts_captura FROM odds_snapshots s "
                    "WHERE match_id = ? AND mercado = ? AND IFNULL(linea,-999) = ? "
                    "  AND ts_captura < ? "
                    "  AND snapshot_id = (SELECT MAX(snapshot_id) FROM odds_snapshots "
                    "                     WHERE match_id = s.match_id AND mercado = s.mercado "
                    "                       AND IFNULL(linea,-999) = IFNULL(s.linea,-999) "
                    "                       AND seleccion = s.seleccion AND ts_captura < ?)",
                    (m["match_id"], mk["mercado"], mk["lin"], m["fecha_evento"],
                     m["fecha_evento"])).fetchall()
                if len(filas) < 2:
                    continue              # mercado incompleto: no se puede neutralizar

                cuotas = [f["cuota"] for f in filas]
                try:
                    p_close = devig(cuotas, method=devig_method)
                except ValueError as exc:
                    log.warning("Cierre invalido en %s/%s: %s", m["match_id"], mk["mercado"], exc)
                    continue
                mapa = {f["seleccion"]: (f["cuota"], float(p))
                        for f, p in zip(filas, p_close)}

                for f in filas:
                    conn.execute("UPDATE odds_snapshots SET es_cierre = 1 WHERE snapshot_id = ?",
                                 (f["snapshot_id"],))
                    n_cierres += 1

                apuestas = conn.execute(
                    "SELECT id_apuesta, seleccion, cuota_betplay FROM bets "
                    "WHERE match_id = ? AND mercado = ? AND IFNULL(linea,-999) = ? "
                    "  AND cuota_cierre IS NULL",
                    (m["match_id"], mk["mercado"], mk["lin"])).fetchall()
                for b in apuestas:
                    if b["seleccion"] not in mapa:
                        continue
                    c_close, p_c = mapa[b["seleccion"]]
                    db.settle_bet(conn, b["id_apuesta"],
                                  cuota_cierre=c_close, p_cierre_devig=p_c,
                                  clv_odds=clv_odds_ratio(b["cuota_betplay"], c_close),
                                  clv_prob=p_c * b["cuota_betplay"] - 1.0)
                    n_clv += 1
        conn.execute("COMMIT")
    log.info("Cierre consolidado: %d snapshots marcados, %d apuestas con CLV.",
             n_cierres, n_clv)
    return {"snapshots_cierre": n_cierres, "apuestas_con_clv": n_clv}


# --------------------------------------------------------------------------- #
# 3. Vigilancia de salud (rupturas silenciosas del extractor)                  #
# --------------------------------------------------------------------------- #
def health_check(db_path: str, fuente: str, n_filas: int,
                 par: CaptureParams) -> dict[str, Any]:
    """Compara el volumen capturado con la mediana movil reciente.

    Cero filas, o una caida por debajo de `salud_umbral` veces la mediana, indica
    casi con seguridad un cambio de DOM o de endpoint, no una jornada tranquila.
    """
    with db.connect(db_path) as conn:
        conn.executescript(SCHEMA_CAPTURA)
        hist = [r["n_filas"] for r in conn.execute(
            "SELECT n_filas FROM health_log WHERE fuente = ? AND estado = 'ok' "
            "ORDER BY ts DESC LIMIT 30", (fuente,)).fetchall()]

        mediana = statistics.median(hist) if len(hist) >= par.salud_min_muestras else None
        if n_filas == 0:
            estado, detalle = "critico", "Cero filas: extractor probablemente roto."
        elif mediana is not None and n_filas < par.salud_umbral * mediana:
            estado = "alerta"
            detalle = (f"Volumen {n_filas} por debajo del {par.salud_umbral:.0%} "
                       f"de la mediana movil ({mediana:.0f}).")
        else:
            estado, detalle = "ok", ""

        conn.execute("INSERT INTO health_log (ts, fuente, n_filas, mediana_ref, estado, detalle) "
                     "VALUES (?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      fuente, n_filas, mediana, estado, detalle))
    if estado != "ok":
        log.error("SALUD [%s] %s: %s", fuente, estado.upper(), detalle)
    return {"fuente": fuente, "n_filas": n_filas, "mediana_ref": mediana,
            "estado": estado, "detalle": detalle}


# --------------------------------------------------------------------------- #
# 4. Ciclo de captura continua                                                 #
# --------------------------------------------------------------------------- #
def run_capture_cycle(cfg: dict[str, Any], params: Any, par: CaptureParams,
                      dry_run: bool = False,
                      ahora: datetime | None = None) -> dict[str, Any]:
    """Un tick del capturador. Pensado para invocarse por cron cada 5 minutos.

    Secuencia:
      1. Captura el arbol de cuotas del horizonte y persiste los snapshots.
      2. Registra que ventanas de la escalera quedan cubiertas y cuales perdidas.
      3. Evalua el filtro +EV y registra apuestas SOLO para los partidos que se
         encuentran en la ventana de colocacion pre-registrada.
      4. Consolida lineas de cierre y CLV de los partidos ya comenzados.
      5. Ejecuta la vigilancia de salud del extractor.

    Es idempotente: reejecutarlo dentro de la misma ventana no duplica snapshots
    de ventana ni apuestas.
    """
    from .pipeline import (agrupar_mercados, indexar_tendencias,
                           evaluar_selecciones, persistir_crudos, _log_decisiones)
    from .scrapers.betplay import BetPlayScraper
    from .scrapers.linemate import LinemateScraper

    ahora = ahora or datetime.now(timezone.utc)
    db_path = cfg.get("storage", {}).get("db_path", str(db.DEFAULT_DB))
    temporada = cfg.get("estudio", {}).get("temporada", "2025-2026")
    run_id = db.new_run_id()
    db.init_db(db_path)
    init_capture_schema(db_path)

    fx = cfg.get("fixtures", {}) if dry_run else {}

    # 1) Barrido del horizonte: se consulta cada dia con partidos programados.
    dias = sorted({(ahora + timedelta(hours=h)).date().isoformat()
                   for h in range(0, int(par.horizonte_h) + 24, 24)})
    odds: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    bp = BetPlayScraper(cfg["betplay"], fixture=fx.get("betplay"))
    lm = LinemateScraper(cfg["linemate"], fixture=fx.get("linemate"))
    for dia in dias:
        odds.extend(bp.fetch(dia))
        trends.extend(lm.fetch(dia))
        if dry_run:
            break                         # el fixture no depende de la fecha

    # Los extractores sellan con el reloj de pared. Se resella con el instante del
    # tick y se recalcula la distancia al evento: `horas_al_inicio` es una
    # covariable del analisis de convergencia de la linea, de modo que debe medir
    # la distancia real entre la captura y el pitido, no el momento del proceso.
    ts_tick = ahora.isoformat(timespec="seconds")
    vivos: list[dict[str, Any]] = []
    for o in odds:
        h = horas_al_inicio(o.get("fecha_evento", ""), ahora)
        # Se descartan los precios posteriores al pitido inicial: un precio en
        # vivo no es comparable con uno pre-partido y, si se colara en la tabla,
        # podria tomarse por linea de cierre.
        if h is not None and h < -TOLERANCIA_H:
            continue
        o["ts_captura"] = ts_tick
        if h is not None:
            o["horas_al_inicio"] = h
        vivos.append(o)
    n_descartados = len(odds) - len(vivos)
    if n_descartados:
        log.info("Descartados %d precios de partidos ya iniciados.", n_descartados)
    odds = vivos
    for t in trends:
        t["ts_captura"] = ts_tick

    salud = [health_check(db_path, "BetPlay", len(odds), par),
             health_check(db_path, "Linemate", len(trends), par)]

    with db.connect(db_path) as conn:
        conn.execute("BEGIN")
        persistir_crudos(conn, odds, trends, temporada, run_id)
        conn.execute("COMMIT")

    # 2) Escalera de ventanas: que hitos cubre esta ejecucion.
    vigilados = partidos_vigilados(db_path, par, ahora)
    cubiertas = perdidas = 0
    en_apuesta: set[str] = set()
    for m in vigilados:
        if m["en_ventana_apuesta"]:
            en_apuesta.add(m["match_id"])
        n_filas = sum(1 for o in odds if o["match_key"] == m["match_id"])
        if m["ventana_debida"] is not None and n_filas:
            registrar_captura(db_path, m["match_id"], m["ventana_debida"],
                              "BetPlay", n_filas, run_id)
            cubiertas += 1
            for v in m["ventanas_perdidas"]:
                registrar_captura(db_path, m["match_id"], v, "BetPlay", -1, run_id)
                perdidas += 1

    # 3) Evaluacion y registro, restringido a la ventana de colocacion.
    with db.connect(db_path) as conn:
        conn.execute("BEGIN")
        decisiones, n_bets = evaluar_selecciones(
            conn, agrupar_mercados(odds), indexar_tendencias(trends),
            params, cfg, run_id, solo_match_ids=en_apuesta)
        conn.execute("COMMIT")
    _log_decisiones(decisiones, run_id, cfg)

    # 4) Cierre y CLV de los partidos ya iniciados.
    cierre = finalize_closing_lines(db_path, ahora, params.devig_method)

    resumen = {
        "run_id": run_id, "ts": ahora.isoformat(timespec="seconds"),
        "n_cuotas": len(odds), "n_tendencias": len(trends),
        "partidos_vigilados": len(vigilados),
        "precios_descartados_en_vivo": n_descartados,
        "ventanas_cubiertas": cubiertas, "ventanas_perdidas": perdidas,
        "partidos_en_ventana_apuesta": len(en_apuesta),
        "n_evaluadas": len(decisiones), "n_apuestas_registradas": n_bets,
        **cierre,
        "salud": salud,
        "estado": "critico" if any(s["estado"] == "critico" for s in salud)
                  else ("alerta" if any(s["estado"] == "alerta" for s in salud) else "ok"),
    }
    log.info("Tick de captura: %s", {k: v for k, v in resumen.items() if k != "salud"})
    return resumen
