"""
model.py — Construccion de la probabilidad estimada p_est a partir de la senal.

Este modulo materializa la **hipotesis de identificacion** del estudio: para que
el filtro +EV no sea tautologico, `p_est` debe incorporar informacion ausente (o
mal ponderada) en el precio de BetPlay. Se implementan tres estimadores, en orden
creciente de exigencia informacional:

  A) `p_shrunk`  — encogimiento empirico-bayesiano de la tendencia de Linemate
     hacia el precio neutralizado de la casa. Es el estimador **conservador**
     por defecto: si la tendencia no es informativa, p_est -> p_mercado y el
     EV colapsa a cero, que es exactamente el comportamiento deseado bajo la EMH.

  B) `p_count`   — modelo generativo Poisson / Binomial Negativa para mercados de
     conteo (corners, tarjetas), calibrado con la media y varianza recientes.
     Constituye una fuente de informacion estructuralmente independiente del
     precio.

  C) `p_blend`   — combinacion log-lineal (*logarithmic opinion pool*) de A y B,
     con pesos fijados ex-ante y registrados en `model_version`.

Nota critica sobre el sesgo de sobreajuste
------------------------------------------
Los pesos y el parametro de encogimiento k NO deben elegirse mirando el PnL de la
muestra de evaluacion: eso induciria *data snooping* y invalidaria el contraste de
hipotesis. El protocolo (ver docs/PREREGISTRO.md) los fija en una muestra de
calibracion previa y los congela antes de la primera apuesta registrada.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .oddsmath import (devig, expected_value, inv_logit, kelly_fraction, logit,
                       negbin_over_prob, poisson_over_prob, shrink_to_prior)

MODEL_VERSION = "v1.0.0-shrunkNB"


@dataclass
class ModelParams:
    """Hiperparametros congelados ex-ante (pre-registro)."""
    k_shrink: float = 12.0          # fuerza del encogimiento hacia el mercado
    w_trend: float = 0.5            # peso de la senal de tendencia en el pool
    w_count: float = 0.5            # peso del modelo de conteo
    devig_method: str = "shin"      # estimador de probabilidad neutralizada
    ev_min: float = 0.03            # umbral minimo de EV para registrar apuesta
    trend_z_min: float = 1.0        # exigencia minima de senal (|z|)
    odds_min: float = 1.40          # evita cuotas muy cortas (ruido de liquidacion)
    odds_max: float = 6.00          # evita longshots (varianza y sesgo F-L)
    max_bets_per_match: int = 2     # limita la dependencia intra-partido
    version: str = MODEL_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


def p_shrunk(tasa_bruta: float, n: int, p_mercado: float, k: float = 12.0) -> float:
    """Estimador A: tendencia encogida hacia el precio neutralizado."""
    return shrink_to_prior(tasa_bruta, n, p_mercado, k=k)


def p_count(media: float | None, varianza: float | None, linea: float | None,
            seleccion: str) -> float | None:
    """Estimador B: modelo de conteo (NB si hay sobredispersion, Poisson si no)."""
    if media is None or linea is None or media <= 0:
        return None
    p_over = (negbin_over_prob(media, varianza, linea) if varianza and varianza > media
              else poisson_over_prob(media, linea))
    return p_over if seleccion == "over" else 1.0 - p_over


def p_blend(p_a: float, p_b: float | None, w_a: float = 0.5, w_b: float = 0.5) -> float:
    r"""Pool logaritmico de opiniones (combinacion lineal en el espacio logit).

    .. math:: \mathrm{logit}(\bar p) = \frac{w_a\,\mathrm{logit}(p_a)
              + w_b\,\mathrm{logit}(p_b)}{w_a + w_b}

    Se prefiere al promedio aritmetico porque preserva mejor la calibracion en las
    colas y es la regla de agregacion externamente bayesiana estandar.
    """
    if p_b is None:
        return p_a
    z = (w_a * logit(p_a) + w_b * logit(p_b)) / (w_a + w_b)
    return float(np.clip(inv_logit(z), 1e-6, 1 - 1e-6))


def estimate_probability(trend: dict[str, Any], p_mercado: float,
                         params: ModelParams) -> tuple[float, dict[str, Any]]:
    """Devuelve (p_est, diagnostico) combinando los estimadores disponibles."""
    p_a = p_shrunk(float(trend["tasa_bruta"]), int(trend["n_muestra"]), p_mercado,
                   k=params.k_shrink)
    p_b = p_count(trend.get("media_reciente"), trend.get("varianza_reciente"),
                  trend.get("linea"), str(trend.get("seleccion", "over")))
    p_est = p_blend(p_a, p_b, params.w_trend, params.w_count)
    return p_est, {"p_shrunk": p_a, "p_count": p_b, "p_mercado": p_mercado}


def market_devig(cuotas_mercado: list[float], idx: int, method: str) -> float:
    """Probabilidad neutralizada de la seleccion `idx` dentro de su mercado."""
    return float(devig(cuotas_mercado, method=method)[idx])


def passes_filter(p_est: float, cuota: float, trend_score: float,
                  params: ModelParams) -> tuple[bool, str]:
    r"""Regla de decision ex-ante. Devuelve (acepta, motivo_rechazo).

    Un apostador solo debe actuar si :math:`\mathrm{EV} = p\,c - 1 > \tau`, con
    :math:`\tau > 0` estrictamente positivo. El umbral `ev_min` no es arbitrario:
    absorbe el **error de estimacion** de p_est. Con tau=0 se apostaria cada vez
    que el ruido de estimacion excede el margen, generando una seleccion adversa
    sistematica (la *maldicion del ganador* aplicada a la seleccion de apuestas).
    """
    if not (params.odds_min <= cuota <= params.odds_max):
        return False, "fuera_rango_cuota"
    if abs(trend_score) < params.trend_z_min:
        return False, "senal_insuficiente"
    ev = expected_value(p_est, cuota)
    if ev <= params.ev_min:
        return False, "ev_bajo_umbral"
    return True, "aceptada"


def build_candidate(trend: dict[str, Any], cuota: float, cuotas_mercado: list[float],
                    idx: int, trend_score: float,
                    params: ModelParams) -> dict[str, Any]:
    """Ensambla el registro ex-ante completo de una apuesta candidata."""
    p_mercado = market_devig(cuotas_mercado, idx, params.devig_method)
    p_est, diag = estimate_probability(trend, p_mercado, params)
    ev = expected_value(p_est, cuota)
    ok, motivo = passes_filter(p_est, cuota, trend_score, params)
    return {
        "p_implicita": 1.0 / cuota,
        "overround": float(np.sum(1.0 / np.asarray(cuotas_mercado)) - 1.0),
        "p_mercado_devig": p_mercado,
        "p_estimada": p_est,
        "ev_calculado": ev,
        "edge_prob": p_est - p_mercado,
        "trend_score": trend_score,
        "kelly_sugerido": kelly_fraction(p_est, cuota, fraction=0.25) if ev > 0 else 0.0,
        "devig_method": params.devig_method,
        "model_version": params.version,
        "aceptada": ok,
        "motivo": motivo,
        "diagnostico": diag,
    }
