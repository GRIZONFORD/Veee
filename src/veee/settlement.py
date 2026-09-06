"""
settlement.py — Liquidacion ex-post: linea de cierre, resultado y PnL.

Reglas de liquidacion (stake plano s = 1 unidad)
------------------------------------------------
.. math::
    \\mathrm{PnL}_i = \\begin{cases}
        s\\,(c_i - 1) & \\text{acierto} \\\\
        -s            & \\text{fallo} \\\\
        0             & \\text{nula (push)}
    \\end{cases}

Las lineas asiaticas de cuarto (p. ej. -0.25, +0.75) dividen el stake en dos
mitades, generando medio acierto o media devolucion; se implementa explicitamente
porque tratarlas como binarias sesga tanto el PnL como la variable dependiente del
logit. Las apuestas nulas se excluyen del logit (y no esta definida) pero se
conservan en el ROI con PnL = 0.

El CLV se calcula contra la **linea de cierre neutralizada**, no contra la cuota
de cierre bruta: comparar una cuota con margen frente a otra con margen confunde
ventaja informativa con variacion del propio margen.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable

from . import database as db
from .oddsmath import clv_odds_ratio, devig

log = logging.getLogger(__name__)


def pnl_binario(resultado: str, cuota: float, stake: float = 1.0) -> float:
    """PnL de un mercado binario o categorico simple."""
    if resultado == "ganada":
        return stake * (cuota - 1.0)
    if resultado == "perdida":
        return -stake
    if resultado in {"nula", "push", "anulada"}:
        return 0.0
    raise ValueError(f"Resultado no reconocido: {resultado!r}")


def liquidar_ou(valor_observado: float, linea: float, seleccion: str) -> str:
    """Liquida un Over/Under de conteo (corners, tarjetas, disparos)."""
    if math.isclose(valor_observado, linea):
        return "nula"
    over = valor_observado > linea
    gana = over if seleccion == "over" else not over
    return "ganada" if gana else "perdida"


def liquidar_ah(margen_goles: float, linea: float, stake: float = 1.0) -> tuple[str, float]:
    r"""Liquida un handicap asiatico. `margen_goles` es (goles equipo - rival).

    Devuelve (etiqueta, pnl_multiplicador_sobre_(c-1)_o_stake). Las lineas de
    cuarto se descomponen en dos medias apuestas sobre las lineas adyacentes.
    """
    ajustado = margen_goles + linea
    if abs(linea * 4 - round(linea * 4)) > 1e-9:
        raise ValueError("Linea asiatica no valida.")
    cuarto = abs(linea * 4) % 2 == 1        # .25 o .75
    if cuarto:
        l1, l2 = linea - 0.25, linea + 0.25
        e1, f1 = liquidar_ah(margen_goles, l1, stake / 2)
        e2, f2 = liquidar_ah(margen_goles, l2, stake / 2)
        etiqueta = "ganada" if f1 + f2 > 0 else ("perdida" if f1 + f2 < 0 else "nula")
        return etiqueta, f1 + f2
    if ajustado > 0:
        return "ganada", stake
    if ajustado < 0:
        return "perdida", -stake
    return "nula", 0.0


def pnl_ah(margen_goles: float, linea: float, cuota: float, stake: float = 1.0) -> tuple[str, float]:
    """PnL de un handicap asiatico incluyendo medias victorias/derrotas."""
    etiqueta, frac = liquidar_ah(margen_goles, linea, stake)
    if frac > 0:
        return etiqueta, frac * (cuota - 1.0)
    return etiqueta, frac


def capturar_cierre(db_path: str, match_id: str, mercado: str, linea: float | None,
                    cuotas_cierre: dict[str, float], run_id: str) -> None:
    """Registra el snapshot de cierre y actualiza el CLV de las apuestas del mercado."""
    selecciones = list(cuotas_cierre)
    p_close = devig([cuotas_cierre[s] for s in selecciones], method="shin")
    mapa = dict(zip(selecciones, p_close))
    with db.connect(db_path) as conn:
        for s, c in cuotas_cierre.items():
            db.insert_snapshot(conn, {
                "match_id": match_id, "casa": "BetPlay", "mercado": mercado,
                "seleccion": s, "linea": linea, "cuota": c,
                "ts_captura": db.datetime.now(db.timezone.utc).isoformat(timespec="seconds"),
                "horas_al_inicio": 0.0, "es_cierre": 1, "run_id": run_id,
            })
        rows = conn.execute(
            "SELECT id_apuesta, seleccion, cuota_betplay FROM bets "
            "WHERE match_id=? AND mercado=? AND IFNULL(linea,-999)=IFNULL(?,-999)",
            (match_id, mercado, linea),
        ).fetchall()
        for r in rows:
            if r["seleccion"] not in cuotas_cierre:
                continue
            c_close = cuotas_cierre[r["seleccion"]]
            db.settle_bet(conn, r["id_apuesta"],
                          cuota_cierre=c_close,
                          p_cierre_devig=float(mapa[r["seleccion"]]),
                          clv_odds=clv_odds_ratio(r["cuota_betplay"], c_close),
                          clv_prob=float(mapa[r["seleccion"]]) * r["cuota_betplay"] - 1.0)


def liquidar_partido(db_path: str, match_id: str,
                     resultados: dict[str, Any]) -> int:
    r"""Liquida todas las apuestas de un partido.

    `resultados` contiene las variables observadas, p. ej.::

        {"corners_totales": 11, "tarjetas_totales": 5,
         "goles_local": 2, "goles_visitante": 1, "resultado_1x2": "home"}
    """
    n = 0
    with db.connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.execute("UPDATE matches SET estado='jugado' WHERE match_id=?", (match_id,))
        for r in conn.execute(
            "SELECT * FROM bets WHERE match_id=? AND resultado_real IS NULL", (match_id,)
        ).fetchall():
            etiqueta, pnl, valor = _liquidar_fila(dict(r), resultados)
            if etiqueta is None:
                continue
            db.settle_bet(conn, r["id_apuesta"], resultado_real=etiqueta,
                          valor_observado=valor, pnl_unidades=pnl,
                          y_exito=None if etiqueta == "nula" else int(etiqueta == "ganada"))
            n += 1
        conn.execute("COMMIT")
    log.info("Liquidadas %d apuestas del partido %s.", n, match_id)
    return n


def _liquidar_fila(bet: dict[str, Any], res: dict[str, Any]):
    """Aplica la regla de liquidacion correspondiente al mercado de la apuesta."""
    mercado, sel, linea = bet["mercado"], bet["seleccion"], bet["linea"]
    cuota, stake = bet["cuota_betplay"], bet["stake"]
    if mercado == "corners_ou" and "corners_totales" in res:
        v = float(res["corners_totales"])
        et = liquidar_ou(v, float(linea), sel)
        return et, pnl_binario(et, cuota, stake), v
    if mercado == "cards_ou" and "tarjetas_totales" in res:
        v = float(res["tarjetas_totales"])
        et = liquidar_ou(v, float(linea), sel)
        return et, pnl_binario(et, cuota, stake), v
    if mercado == "1X2" and "resultado_1x2" in res:
        et = "ganada" if sel == res["resultado_1x2"] else "perdida"
        return et, pnl_binario(et, cuota, stake), None
    if mercado == "ah" and {"goles_local", "goles_visitante"} <= res.keys():
        margen = float(res["goles_local"]) - float(res["goles_visitante"])
        if sel in {"away", "visitante"}:
            margen = -margen
        et, pnl = pnl_ah(margen, float(linea), cuota, stake)
        return et, pnl, margen
    if mercado == "prop_player" and sel in res:
        v = float(res[sel])
        et = liquidar_ou(v, float(linea), "over")
        return et, pnl_binario(et, cuota, stake), v
    return None, None, None


def liquidar_lote(db_path: str, partidos: Iterable[dict[str, Any]]) -> int:
    """Liquida una coleccion de partidos: [{'match_id':..., 'resultados': {...}}, ...]."""
    return sum(liquidar_partido(db_path, p["match_id"], p["resultados"]) for p in partidos)
