r"""
simulate.py — Proceso generador de datos (DGP) sintetico para validacion del diseno.

Proposito
---------
Permite (i) validar el pipeline completo antes de recolectar un solo dato real,
(ii) verificar que el contraste tiene el **tamano nominal** bajo la hipotesis nula
(rechaza aproximadamente el 5% de las veces cuando el mercado es eficiente) y
(iii) calcular por simulacion la **potencia** frente a alternativas realistas. Es
el equivalente computacional de un experimento Monte Carlo de pre-registro.

Descomposicion del DGP
----------------------
La probabilidad real se descompone en un componente estructural (calidad de los
equipos, de dominio publico) y un componente de **forma reciente**, que es
justamente lo que las tendencias de Linemate revelan:

.. math:: \mathrm{logit}(p^{true}) = \eta_{base} + \delta

La casa cotiza :math:`\eta_{base} + \lambda\,\delta + \nu`, donde :math:`\lambda`
mide cuanta de esa informacion incorpora al precio y :math:`\nu` es ruido
idiosincratico de fijacion de precios.

Escenarios
----------
* ``escenario_h0()`` — mercado eficiente en sentido semifuerte: :math:`\lambda=1`
  y :math:`\nu \equiv 0`. El precio es estadistico suficiente de la informacion
  publica y NINGUNA estrategia puede tener EV positivo: el retorno esperado de
  toda apuesta es exactamente :math:`-m/(1+m)`, el margen normalizado. Es el
  patron de referencia para verificar el tamano del contraste.
* ``escenario_h1_subreaccion()`` — :math:`\lambda<1`: la casa **subreacciona** a
  la forma reciente. Es la ineficiencia conductual que el estudio busca detectar,
  analoga a la subreaccion documentada en mercados financieros por Bernard &
  Thomas (1989).
* ``escenario_h1_ruido()`` — :math:`\lambda=1` pero :math:`\nu>0`. Punto
  metodologico relevante: el ruido de fijacion de precios es explotable por si
  solo por cualquier agente con informacion independiente sobre :math:`p^{true}`,
  aunque la casa no cometa ningun sesgo sistematico. Confundir ambos escenarios
  llevaria a atribuir a subreaccion lo que es mera dispersion de precios.

Parametros calibrados con ordenes de magnitud realistas para LaLiga: margen del
5% en 1X2 y 7% en mercados de conteo; ventanas de tendencia de 10 partidos.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import database as db
from .model import ModelParams, build_candidate
from .oddsmath import devig, inv_logit, logit
from .scrapers.linemate import trend_z
from .settlement import pnl_binario


@dataclass
class DGPParams:
    """Parametros del proceso generador."""
    n_dias: int = 120
    partidos_por_dia: int = 4
    mercados_por_partido: int = 3
    vig: float = 0.06               # margen bruto de la casa
    n_ventana: int = 10             # tamano de la ventana de tendencia
    sigma_base: float = 0.50        # dispersion del componente estructural
    sigma_delta: float = 0.35       # dispersion del componente de forma reciente
    sigma_precio: float = 0.0       # ruido idiosincratico de precio (nu)
    lam: float = 1.0                # fraccion de delta incorporada al precio
    sigma_cierre: float = 0.10      # ruido residual de la linea de cierre
    seed: int = 20260101


def escenario_h0(**kw) -> DGPParams:
    """Mercado eficiente: el precio agota la informacion publica (lambda=1, nu=0)."""
    return DGPParams(**{**{"lam": 1.0, "sigma_precio": 0.0}, **kw})


def escenario_h1_subreaccion(lam: float = 0.6, **kw) -> DGPParams:
    """Subreaccion de la casa a la forma reciente: ineficiencia conductual."""
    return DGPParams(**{**{"lam": lam, "sigma_precio": 0.0}, **kw})


def escenario_h1_ruido(sigma_precio: float = 0.30, **kw) -> DGPParams:
    """Precio insesgado pero disperso: ineficiencia por ruido, no por sesgo."""
    return DGPParams(**{**{"lam": 1.0, "sigma_precio": sigma_precio}, **kw})


def roi_teorico_h0(vig: float) -> float:
    r"""Retorno esperado por apuesta bajo eficiencia: :math:`-m/(1+m)`."""
    return -vig / (1.0 + vig)


def simular_muestra(par: DGPParams, params: ModelParams) -> pd.DataFrame:
    r"""Genera una muestra de apuestas ya liquidadas segun el DGP.

    .. math::
        \mathrm{logit}(p^{true}) = \eta_{base} + \delta

        \mathrm{logit}(p^{book}) = \eta_{base} + \lambda\,\delta + \nu,
        \qquad \nu \sim N(0, \sigma_{\nu}^2)

    Con :math:`\lambda=1` y :math:`\nu \equiv 0` el precio es estadistico
    suficiente y el retorno esperado de cualquier apuesta es :math:`-m/(1+m)`.
    Con :math:`\lambda<1` el precio subreacciona a la forma reciente. La tendencia
    se genera como :math:`\text{hits} \sim \mathrm{Bin}(n, p^{true})`: informativa
    sobre :math:`p^{true}`, pero ruidosa.
    """
    rng = np.random.default_rng(par.seed)
    filas: list[dict[str, Any]] = []
    d0 = date(2025, 8, 15)
    mercados = ["corners_ou", "cards_ou", "ah"]
    factor = 1.0 / (1.0 + par.vig)          # reparto proporcional del margen

    for dia in range(par.n_dias):
        fecha = (d0 + timedelta(days=dia)).isoformat()
        for j in range(par.partidos_por_dia):
            mid = f"sim_{dia:03d}_{j}"
            partido = f"Equipo{2 * j} vs Equipo{2 * j + 1}"
            for k in range(par.mercados_por_partido):
                mercado = mercados[k % len(mercados)]

                # Componente estructural (publico) + forma reciente (revelada por
                # la tendencia). La casa incorpora la segunda con intensidad lambda.
                eta_base = rng.normal(0.0, par.sigma_base)
                delta = rng.normal(0.0, par.sigma_delta)
                p_true = float(np.clip(inv_logit(eta_base + delta), 0.10, 0.90))

                nu = rng.normal(0.0, par.sigma_precio) if par.sigma_precio > 0 else 0.0
                p_book = float(np.clip(inv_logit(eta_base + par.lam * delta + nu),
                                       0.06, 0.94))

                cuotas = [round(factor / p_book, 2), round(factor / (1.0 - p_book), 2)]
                if min(cuotas) < 1.02:
                    continue

                # Tendencia de Linemate: frecuencia binomial sobre la ventana movil.
                hits = int(rng.binomial(par.n_ventana, p_true))
                trend = {"tasa_bruta": hits / par.n_ventana, "n_muestra": par.n_ventana,
                         "linea": 9.5, "seleccion": "over",
                         "media_reciente": None, "varianza_reciente": None}

                p_mkt = float(devig(cuotas, method=params.devig_method)[0])
                z = trend_z(trend["tasa_bruta"], par.n_ventana, p_mkt)
                cand = build_candidate(trend, cuotas[0], cuotas, 0, z, params)
                if not cand["aceptada"]:
                    continue

                # Linea de cierre: converge hacia p_true con ruido residual menor
                # (supuesto estandar: el cierre es el precio mas eficiente).
                p_close = float(np.clip(inv_logit(logit(p_true) +
                                                  rng.normal(0.0, par.sigma_cierre)),
                                        0.05, 0.95))
                c_close = max(round(factor / p_close, 2), 1.01)

                gana = bool(rng.random() < p_true)
                etiqueta = "ganada" if gana else "perdida"

                filas.append({
                    "id_apuesta": db.bet_id(mid, mercado, "over", 9.5, fecha),
                    "fecha": fecha, "fecha_evento": f"{fecha}T19:00:00+00:00",
                    "match_id": mid, "partido": partido, "mercado": mercado,
                    "seleccion": "over", "linea": 9.5, "casa": "BetPlay",
                    "cuota_betplay": cuotas[0], "p_implicita": cand["p_implicita"],
                    "overround": cand["overround"],
                    "p_mercado_devig": cand["p_mercado_devig"],
                    "devig_method": cand["devig_method"],
                    "trend_score": z, "p_estimada": cand["p_estimada"],
                    "ev_calculado": cand["ev_calculado"], "edge_prob": cand["edge_prob"],
                    "stake": 1.0, "kelly_sugerido": cand["kelly_sugerido"],
                    "model_version": cand["model_version"],
                    "ts_colocacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "horas_al_inicio": 6.0,
                    "cuota_cierre": c_close, "p_cierre_devig": p_close,
                    "clv_odds": cuotas[0] / c_close - 1.0,
                    "clv_prob": p_close * cuotas[0] - 1.0,
                    "resultado_real": etiqueta, "valor_observado": None,
                    "y_exito": int(gana),
                    "pnl_unidades": pnl_binario(etiqueta, cuotas[0], 1.0),
                    "ts_liquidacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "p_true": p_true,
                })
    return pd.DataFrame(filas)


def monte_carlo(par: DGPParams, params: ModelParams, n_rep: int = 200,
                alpha: float = 0.05) -> dict[str, float]:
    """Tasa de rechazo empirica del contraste t unilateral sobre el ROI diario.

    Bajo `escenario_h0` debe aproximarse a `alpha`: verifica que el procedimiento
    inferencial no sobre-rechaza. Bajo los escenarios H1 estima la **potencia**.
    """
    from .econometrics import daily_roi, prepare, t_test_roi
    rechazos, rois, n_obs = 0, [], []
    for r in range(n_rep):
        p = DGPParams(**{**par.__dict__, "seed": par.seed + r})
        df = simular_muestra(p, params)
        if df.empty or df["fecha"].nunique() < 3:
            continue
        d = prepare(df)
        rois.append(float(d["pnl_unidades"].sum() / d["stake"].sum()))
        n_obs.append(len(d))
        if t_test_roi(daily_roi(d)["roi"].values).p_valor < alpha:
            rechazos += 1
    return {"n_rep": n_rep, "tasa_rechazo": rechazos / max(n_rep, 1),
            "roi_medio": float(np.mean(rois)) if rois else float("nan"),
            "n_apuestas_medio": float(np.mean(n_obs)) if n_obs else float("nan"),
            "lambda": par.lam, "sigma_precio": par.sigma_precio, "alpha": alpha}
