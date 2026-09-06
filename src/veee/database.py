"""
database.py — Esquema y capa de persistencia (SQLite) del registro de paper trading.

Principios de diseno
--------------------
1. **Inmutabilidad del registro ex-ante.** Una fila de `bets` se escribe *antes*
   del inicio del partido y sus campos ex-ante (cuota, EV, trend_score) nunca se
   reescriben. La liquidacion solo puebla campos ex-post. Esto elimina por
   construccion el sesgo de *look-ahead* y hace el estudio auditable.
2. **Trazabilidad.** Cada fila guarda `model_version`, `devig_method` y `run_id`,
   de modo que cualquier resultado econometrico sea reproducible bit a bit.
3. **Separacion snapshot / apuesta.** `odds_snapshots` conserva el historico
   completo de precios (apertura -> cierre), lo que permite reconstruir el CLV y
   estudiar la dinamica de convergencia de la linea sin contaminar `bets`.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

DEFAULT_DB = Path("data/veee.db")

SCHEMA = """
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- partidos --
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,          -- hash estable: liga|fecha|local|visitante
    liga            TEXT NOT NULL DEFAULT 'LaLiga',
    temporada       TEXT NOT NULL,
    jornada         INTEGER,
    fecha_evento    TEXT NOT NULL,             -- ISO-8601 UTC del pitido inicial
    equipo_local    TEXT NOT NULL,
    equipo_visitante TEXT NOT NULL,
    estado          TEXT NOT NULL DEFAULT 'programado',  -- programado|jugado|suspendido
    created_at      TEXT NOT NULL
);

-- ------------------------------------------------- historico de cotizaciones --
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        TEXT NOT NULL REFERENCES matches(match_id),
    casa            TEXT NOT NULL DEFAULT 'BetPlay',
    mercado         TEXT NOT NULL,             -- 1X2|corners_ou|cards_ou|ah|prop_player
    seleccion       TEXT NOT NULL,             -- home|draw|away|over|under|<jugador>
    linea           REAL,                      -- 9.5, -0.5, ... NULL en 1X2
    cuota           REAL NOT NULL CHECK (cuota >= 1.01),
    ts_captura      TEXT NOT NULL,             -- ISO-8601 UTC
    horas_al_inicio REAL,                      -- distancia temporal al evento
    es_cierre       INTEGER NOT NULL DEFAULT 0,
    run_id          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_snap_match ON odds_snapshots(match_id, mercado, seleccion, ts_captura);

-- ------------------------------------------------ tendencias de Linemate ----
CREATE TABLE IF NOT EXISTS trends (
    trend_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        TEXT NOT NULL REFERENCES matches(match_id),
    fuente          TEXT NOT NULL DEFAULT 'Linemate',
    mercado         TEXT NOT NULL,
    seleccion       TEXT NOT NULL,
    linea           REAL,
    hits            INTEGER,                   -- exitos observados en la ventana
    n_muestra       INTEGER,                   -- tamano de la ventana (p.ej. 10)
    tasa_bruta      REAL,                      -- hits / n_muestra
    media_reciente  REAL,                      -- media de la variable de conteo
    varianza_reciente REAL,                    -- para corregir sobredispersion
    trend_score     REAL,                      -- z-score estandarizado de la senal
    ts_captura      TEXT NOT NULL,
    run_id          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trend_match ON trends(match_id, mercado, seleccion);

-- ---------------------------------------- MATRIZ ECONOMETRICA PRINCIPAL -----
CREATE TABLE IF NOT EXISTS bets (
    -- ......................... identificacion
    id_apuesta      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    fecha           TEXT NOT NULL,             -- fecha de COLOCACION (ex-ante)
    fecha_evento    TEXT NOT NULL,
    match_id        TEXT NOT NULL REFERENCES matches(match_id),
    partido         TEXT NOT NULL,             -- 'Real Madrid vs Girona'
    -- ......................... definicion del mercado
    mercado         TEXT NOT NULL,
    seleccion       TEXT NOT NULL,
    linea           REAL,
    casa            TEXT NOT NULL DEFAULT 'BetPlay',
    -- ......................... variables EX-ANTE (inmutables)
    cuota_betplay   REAL NOT NULL CHECK (cuota_betplay >= 1.01),
    p_implicita     REAL NOT NULL,             -- 1/cuota
    overround       REAL NOT NULL,             -- margen del mercado completo
    p_mercado_devig REAL NOT NULL,             -- precio neutralizado de la casa
    devig_method    TEXT NOT NULL,
    trend_score     REAL NOT NULL,
    p_estimada      REAL NOT NULL,             -- creencia del modelo propio
    ev_calculado    REAL NOT NULL,             -- p_estimada*cuota - 1
    edge_prob       REAL NOT NULL,             -- p_estimada - p_mercado_devig
    stake           REAL NOT NULL DEFAULT 1.0, -- stake plano = 1 unidad
    kelly_sugerido  REAL,                      -- solo robustez, no se ejecuta
    model_version   TEXT NOT NULL,
    ts_colocacion   TEXT NOT NULL,
    horas_al_inicio REAL,
    -- ......................... variables EX-POST (liquidacion)
    cuota_cierre    REAL,
    p_cierre_devig  REAL,
    clv_odds        REAL,                      -- cuota_betplay/cuota_cierre - 1
    clv_prob        REAL,                      -- p_cierre_devig*cuota_betplay - 1
    resultado_real  TEXT,                      -- ganada|perdida|nula(push)|anulada
    valor_observado REAL,                      -- p.ej. corners totales observados
    y_exito         INTEGER,                   -- 1 ganada, 0 perdida (NULL si push)
    pnl_unidades    REAL,                      -- +(c-1)*stake | -stake | 0
    ts_liquidacion  TEXT,
    notas           TEXT
);
CREATE INDEX IF NOT EXISTS ix_bets_fecha ON bets(fecha);
CREATE INDEX IF NOT EXISTS ix_bets_match ON bets(match_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_bets_dedup
    ON bets(match_id, mercado, seleccion, IFNULL(linea,-999), fecha);

-- ------------------------------------------------- vistas de agregacion -----
CREATE VIEW IF NOT EXISTS v_daily_roi AS
SELECT fecha,
       COUNT(*)                                   AS n_apuestas,
       SUM(stake)                                 AS stake_total,
       SUM(pnl_unidades)                          AS pnl_total,
       SUM(pnl_unidades) / NULLIF(SUM(stake), 0)  AS roi_diario,
       AVG(ev_calculado)                          AS ev_medio,
       AVG(clv_prob)                              AS clv_medio
FROM bets
WHERE resultado_real IS NOT NULL AND resultado_real <> 'anulada'
GROUP BY fecha
ORDER BY fecha;

CREATE VIEW IF NOT EXISTS v_por_mercado AS
SELECT mercado,
       COUNT(*)                                   AS n,
       AVG(cuota_betplay)                         AS cuota_media,
       AVG(overround)                             AS vig_medio,
       AVG(ev_calculado)                          AS ev_medio,
       AVG(CAST(y_exito AS REAL))                 AS tasa_acierto,
       SUM(pnl_unidades) / NULLIF(SUM(stake), 0)  AS roi,
       AVG(clv_prob)                              AS clv_medio
FROM bets
WHERE resultado_real IS NOT NULL AND resultado_real <> 'anulada'
GROUP BY mercado;
"""

# Orden canonico de columnas de la matriz para exportacion a CSV / replicacion.
MATRIX_COLUMNS: list[str] = [
    "id_apuesta", "fecha", "partido", "mercado", "seleccion", "linea",
    "trend_score", "cuota_betplay", "p_implicita", "overround", "p_mercado_devig",
    "p_estimada", "ev_calculado", "edge_prob", "stake",
    "cuota_cierre", "p_cierre_devig", "clv_odds", "clv_prob",
    "resultado_real", "y_exito", "pnl_unidades",
]


def new_run_id() -> str:
    """Identificador de ejecucion (trazabilidad de cada corrida del pipeline)."""
    return f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"


def bet_id(match_id: str, mercado: str, seleccion: str, linea: float | None,
           fecha: str) -> str:
    """ID determinista de apuesta: garantiza idempotencia del pipeline diario."""
    raw = f"{match_id}|{mercado}|{seleccion}|{linea}|{fecha}"
    return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]


@contextmanager
def connect(db_path: str | Path = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    """Conexion transaccional con claves foraneas activas y filas tipo dict."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    """Crea (idempotentemente) tablas, indices y vistas."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _upsert(conn: sqlite3.Connection, table: str, row: dict[str, Any],
            conflict: str = "IGNORE") -> None:
    cols = ", ".join(row)
    ph = ", ".join("?" for _ in row)
    conn.execute(f"INSERT OR {conflict} INTO {table} ({cols}) VALUES ({ph})",
                 tuple(row.values()))


def upsert_match(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    _upsert(conn, "matches", row)


def insert_snapshot(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _upsert(conn, "odds_snapshots", row)


def insert_trend(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _upsert(conn, "trends", row)


def insert_bet(conn: sqlite3.Connection, row: dict[str, Any]) -> bool:
    """Inserta una apuesta ex-ante. Devuelve False si ya existia (deduplicacion).

    `INSERT OR IGNORE` silencia por igual los duplicados (comportamiento deseado,
    el pipeline es idempotente) y las violaciones de CHECK. Una cuota corrupta
    desapareceria sin dejar rastro y sesgaria la muestra por omision, de modo que
    se valida explicitamente antes de delegar en SQLite.
    """
    cuota = row.get("cuota_betplay")
    if cuota is None or not (1.01 <= float(cuota) <= 1001.0):
        raise sqlite3.IntegrityError(
            f"Cuota invalida ({cuota}) en la apuesta {row.get('id_apuesta')}: "
            "posible error de extraccion."
        )
    for campo in ("p_estimada", "p_mercado_devig", "p_implicita"):
        v = row.get(campo)
        if v is not None and not (0.0 < float(v) < 1.0):
            raise sqlite3.IntegrityError(f"{campo} fuera de (0,1): {v}")
    before = conn.total_changes
    _upsert(conn, "bets", row, conflict="IGNORE")
    return conn.total_changes > before


def settle_bet(conn: sqlite3.Connection, id_apuesta: str, **fields: Any) -> None:
    """Actualiza EXCLUSIVAMENTE campos ex-post. Rechaza escrituras ex-ante."""
    ex_ante = {"cuota_betplay", "ev_calculado", "trend_score", "p_estimada",
               "p_mercado_devig", "p_implicita", "overround", "stake", "fecha"}
    invalid = ex_ante & fields.keys()
    if invalid:
        raise ValueError(
            f"Intento de reescribir campos ex-ante {sorted(invalid)} en {id_apuesta}: "
            "prohibido por el protocolo anti look-ahead."
        )
    fields.setdefault("ts_liquidacion", datetime.now(timezone.utc).isoformat())
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE bets SET {sets} WHERE id_apuesta = ?",
                 (*fields.values(), id_apuesta))


def fetch_df(db_path: str | Path = DEFAULT_DB, query: str = "SELECT * FROM bets",
             params: Sequence[Any] = ()) -> pd.DataFrame:
    """Extrae una consulta a DataFrame (interfaz para el modulo econometrico)."""
    with connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=list(params))


def export_matrix(db_path: str | Path = DEFAULT_DB,
                  csv_path: str | Path = "outputs/matriz_econometrica.csv") -> Path:
    """Exporta la matriz econometrica canonica a CSV (anexo replicable del paper)."""
    df = fetch_df(db_path, f"SELECT {', '.join(MATRIX_COLUMNS)} FROM bets ORDER BY fecha, id_apuesta")
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    return out
