"""
oddsmath.py — Aritmetica de cuotas, eliminacion del margen (de-vigging) y valor esperado.

Modulo nucleo del proyecto. Toda transformacion cuota -> probabilidad pasa por aqui,
de modo que la definicion operativa de "probabilidad justa" sea unica y auditable.

Convenciones
------------
* Las cuotas son decimales europeas (BetPlay las publica en este formato): c >= 1.01.
* Un "mercado" es el conjunto exhaustivo y mutuamente excluyente de resultados
  (1X2 -> 3 selecciones; Over/Under -> 2 selecciones; Handicap asiatico -> 2).
* q_i = 1/c_i es la probabilidad *implicita bruta*; sum_i q_i = 1 + m, con m el
  overround (vig). p_i denota la probabilidad *neutralizada* (sum_i p_i = 1).

Referencias metodologicas
-------------------------
Shin, H. S. (1993). "Measuring the Incidence of Insider Trading in a Market for
    State-Contingent Claims". Economic Journal 103(420), 1141-1153.
Clarke, S. et al. (2017). "Adjusting Bookmaker's Odds to Allow for Overround".
    American Journal of Sports Science 5(6), 45-49.
Stefan, M. & Stefan, F. (2018) sobre el sesgo favorito-longshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import optimize, stats

__all__ = [
    "implied_prob", "overround", "booksum", "fair_odds",
    "devig_proportional", "devig_additive", "devig_power", "devig_shin", "devig",
    "expected_value", "edge_pct", "breakeven_prob", "kelly_fraction",
    "clv_odds_ratio", "clv_prob", "poisson_over_prob", "negbin_over_prob",
    "shrink_to_prior", "logit", "inv_logit", "MIN_ODDS", "MAX_ODDS",
]

# Guardas de sanidad: cuotas fuera de este rango se consideran errores de scraping.
MIN_ODDS = 1.01
MAX_ODDS = 1001.0
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Probabilidades implicitas y margen                                        #
# --------------------------------------------------------------------------- #
def _as_array(odds: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(odds), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Se espera un vector no vacio de cuotas decimales.")
    if np.any(~np.isfinite(arr)) or np.any(arr < MIN_ODDS) or np.any(arr > MAX_ODDS):
        raise ValueError(f"Cuotas fuera del rango admisible [{MIN_ODDS}, {MAX_ODDS}]: {arr}")
    return arr


def implied_prob(odds: float | Iterable[float]) -> float | np.ndarray:
    r"""Probabilidad implicita bruta: :math:`q_i = 1 / c_i`.

    No es una probabilidad en sentido estricto porque :math:`\sum_i q_i > 1`.
    """
    if np.isscalar(odds):
        o = float(odds)  # type: ignore[arg-type]
        if not (MIN_ODDS <= o <= MAX_ODDS):
            raise ValueError(f"Cuota fuera de rango: {o}")
        return 1.0 / o
    return 1.0 / _as_array(odds)


def booksum(odds: Iterable[float]) -> float:
    r""":math:`\sum_i 1/c_i`, tambien llamado *book percentage*."""
    return float(np.sum(1.0 / _as_array(odds)))


def overround(odds: Iterable[float]) -> float:
    r"""Margen bruto de la casa: :math:`m = \sum_i 1/c_i - 1`.

    Es el impuesto implicito que el apostador paga por transaccion. Para LaLiga en
    mercados 1X2 tipicamente m in [0.04, 0.08]; en *props* de jugador m puede
    superar 0.10, lo que eleva sustancialmente el umbral de rentabilidad.
    """
    return booksum(odds) - 1.0


def fair_odds(p: float | Iterable[float]) -> float | np.ndarray:
    """Cuota justa (actuarialmente neutra) asociada a una probabilidad."""
    if np.isscalar(p):
        pv = float(p)  # type: ignore[arg-type]
        if not (0.0 < pv < 1.0):
            raise ValueError("La probabilidad debe estar en (0,1).")
        return 1.0 / pv
    arr = np.asarray(list(p), dtype=float)
    if np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError("Las probabilidades deben estar en (0,1).")
    return 1.0 / arr


# --------------------------------------------------------------------------- #
# 2. Eliminacion del margen (de-vigging)                                       #
# --------------------------------------------------------------------------- #
def devig_proportional(odds: Iterable[float]) -> np.ndarray:
    r"""Normalizacion multiplicativa (*basic / proportional method*).

    .. math:: p_i = \frac{q_i}{\sum_j q_j}, \qquad q_i = 1/c_i

    Supuesto implicito: la casa aplica el margen de forma **proporcional** a la
    probabilidad de cada seleccion. Es el estimador de referencia solicitado en el
    diseno, pero es sabido que sobreestima la probabilidad de los *longshots*
    cuando existe sesgo favorito-longshot (Stefan & Stefan, 2018). Por ello se
    reporta como especificacion base y los metodos de Shin y potencia como
    robustez.
    """
    q = 1.0 / _as_array(odds)
    return q / q.sum()


def devig_additive(odds: Iterable[float]) -> np.ndarray:
    r"""Metodo aditivo: reparte el exceso de margen por igual entre selecciones.

    .. math:: p_i = q_i - \frac{\sum_j q_j - 1}{n}
    """
    q = 1.0 / _as_array(odds)
    p = q - (q.sum() - 1.0) / q.size
    if np.any(p <= 0):
        # Degenera con cuotas muy dispares; se cae al metodo proporcional.
        return devig_proportional(odds)
    return p


def devig_power(odds: Iterable[float]) -> np.ndarray:
    r"""Metodo de potencia (*power / logarithmic method*).

    Busca :math:`k>0` tal que :math:`\sum_i q_i^{\,k} = 1` y define
    :math:`p_i = q_i^{\,k}`. Al ser una transformacion convexa, castiga
    proporcionalmente mas a las selecciones de cuota alta, corrigiendo
    parcialmente el sesgo favorito-longshot.
    """
    q = 1.0 / _as_array(odds)

    def f(k: float) -> float:
        return float(np.sum(q ** k) - 1.0)

    try:
        k = optimize.brentq(f, 0.2, 10.0, xtol=1e-12, maxiter=200)
    except ValueError:
        return devig_proportional(odds)
    p = q ** k
    return p / p.sum()  # saneamiento numerico


def devig_shin(odds: Iterable[float]) -> np.ndarray:
    r"""Metodo de Shin (1993): margen como compensacion por *insider trading*.

    La casa fija precios frente a una proporcion :math:`z` de apostadores
    informados. Invirtiendo el modelo:

    .. math::
        p_i = \frac{\sqrt{z^2 + 4(1-z)\,\dfrac{q_i^2}{Q}} - z}{2(1-z)},
        \qquad Q=\sum_j q_j

    y se resuelve :math:`z \in [0,1)` por biseccion imponiendo
    :math:`\sum_i p_i = 1`. Interpretacion economica directa: :math:`z` es una
    medida de asimetria de informacion percibida por la casa, y por tanto un
    parametro de interes en si mismo para el marco teorico del estudio.
    """
    q = 1.0 / _as_array(odds)
    Q = q.sum()
    if q.size < 2 or Q <= 1.0 + _EPS:
        return devig_proportional(odds)

    def p_of_z(z: float) -> np.ndarray:
        disc = z ** 2 + 4.0 * (1.0 - z) * (q ** 2) / Q
        return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

    def g(z: float) -> float:
        return float(p_of_z(z).sum() - 1.0)

    lo, hi = 0.0, 0.9999
    if g(lo) * g(hi) > 0:
        return devig_proportional(odds)
    z = optimize.brentq(g, lo, hi, xtol=1e-12, maxiter=200)
    p = p_of_z(z)
    return p / p.sum()


_DEVIG_METHODS = {
    "proportional": devig_proportional,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds: Iterable[float], method: str = "proportional") -> np.ndarray:
    """Punto de entrada unico al de-vigging. `method` in {proportional, additive, power, shin}."""
    try:
        fn = _DEVIG_METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Metodo de de-vig desconocido: {method!r}. "
                         f"Opciones: {sorted(_DEVIG_METHODS)}") from exc
    return fn(odds)


# --------------------------------------------------------------------------- #
# 3. Valor esperado, umbral de rentabilidad y dimensionamiento                 #
# --------------------------------------------------------------------------- #
def expected_value(p_est: float, odds: float, stake: float = 1.0) -> float:
    r"""Valor esperado por unidad apostada.

    .. math:: \mathrm{EV} = p\,(c-1) - (1-p) = p\,c - 1

    ADVERTENCIA METODOLOGICA (identificacion): si `p_est` se obtiene neutralizando
    las cuotas **de la misma casa** sobre la que se apuesta, entonces por
    construccion :math:`\sum_i p_i c_i - 1 \approx 0` y el filtro +EV es vacuo.
    `p_est` debe provenir de un conjunto de informacion *distinto* (tendencias de
    Linemate, modelo Poisson propio, o consenso de casas de referencia). Esta es
    la hipotesis de identificacion central del trabajo.
    """
    if not (0.0 < p_est < 1.0):
        raise ValueError("p_est debe estar en (0,1).")
    if not (MIN_ODDS <= odds <= MAX_ODDS):
        raise ValueError(f"Cuota fuera de rango: {odds}")
    return stake * (p_est * odds - 1.0)


def edge_pct(p_est: float, odds: float) -> float:
    """EV expresado en puntos porcentuales del stake (equivalente al ROI esperado)."""
    return 100.0 * expected_value(p_est, odds)


def breakeven_prob(odds: float) -> float:
    r"""Probabilidad de equilibrio :math:`p^* = 1/c`. Apostar exige :math:`p > p^*`."""
    return float(implied_prob(odds))  # type: ignore[return-value]


def kelly_fraction(p_est: float, odds: float, fraction: float = 1.0,
                   cap: float = 0.05) -> float:
    r"""Criterio de Kelly (opcional, solo para analisis de robustez).

    .. math:: f^* = \frac{p\,(c-1) - (1-p)}{c-1} = \frac{p\,c-1}{c-1}

    El diseno principal usa **stake plano** de 1 unidad para que el ROI sea un
    promedio no ponderado y la inferencia sobre :math:`\mu_{ROI}` no confunda
    habilidad predictiva con gestion de banca. `fraction` permite Kelly
    fraccionario (p. ej. 0.25) y `cap` trunca la exposicion maxima.
    """
    b = odds - 1.0
    f = (p_est * odds - 1.0) / b
    return float(np.clip(f * fraction, 0.0, cap))


# --------------------------------------------------------------------------- #
# 4. Closing Line Value (CLV)                                                  #
# --------------------------------------------------------------------------- #
def clv_odds_ratio(odds_taken: float, odds_close: float) -> float:
    r"""CLV en escala de cuotas: :math:`\mathrm{CLV}^{odds} = c_{tomada}/c_{cierre} - 1`.

    Positivo si se tomo un precio mejor que el de cierre. Bajo la hipotesis de que
    la linea de cierre es el estimador mas eficiente de la probabilidad real
    (Bassett, 1981; Woodland & Woodland, 1994), un CLV medio positivo y
    significativo es evidencia de captura genuina de informacion, y no de suerte.
    """
    return odds_taken / odds_close - 1.0


def clv_prob(odds_taken: float, close_odds_market: Sequence[float],
             index: int = 0, method: str = "proportional") -> float:
    r"""CLV en escala de probabilidad, neutralizando el margen de la linea de cierre.

    .. math:: \mathrm{CLV}^{p} = p^{cierre}_{i}\, c^{tomada}_{i} - 1

    Es el EV evaluado con la probabilidad de cierre neutralizada: mide en unidades
    monetarias por unidad apostada la ventaja capturada frente al mercado maduro.
    """
    p_close = devig(close_odds_market, method=method)[index]
    return float(p_close * odds_taken - 1.0)


# --------------------------------------------------------------------------- #
# 5. Modelos generativos para mercados de conteo (corners, tarjetas)           #
# --------------------------------------------------------------------------- #
def poisson_over_prob(lam: float, line: float) -> float:
    r"""P(X > line) con :math:`X \sim \mathrm{Poisson}(\lambda)`.

    Para lineas semienteras (9.5 corners) no hay empate (*push*):
    :math:`P(X \ge \lceil line \rceil)`.
    """
    if lam <= 0:
        raise ValueError("lambda debe ser positivo.")
    k = int(np.floor(line))
    return float(stats.poisson.sf(k, lam))


def negbin_over_prob(mean: float, var: float, line: float) -> float:
    r"""P(X > line) con Binomial Negativa ajustada por momentos.

    Los corners y tarjetas presentan **sobredispersion** (var > media) por
    correlacion intra-partido (estilo de juego, criterio arbitral). Ignorarla
    sesga a la baja la probabilidad de las colas y por tanto infla artificialmente
    el EV de los *overs* con linea alta.
    """
    if var <= mean:
        return poisson_over_prob(mean, line)
    p = mean / var
    n = mean * p / (1.0 - p)
    k = int(np.floor(line))
    return float(stats.nbinom.sf(k, n, p))


# --------------------------------------------------------------------------- #
# 6. Utilidades de encogimiento (shrinkage) y enlace logistico                 #
# --------------------------------------------------------------------------- #
def shrink_to_prior(p_hat: float, n: int, p_prior: float, k: float = 10.0) -> float:
    r"""Encogimiento empirico-bayesiano de una frecuencia muestral hacia un prior.

    .. math:: \tilde{p} = w\,\hat{p} + (1-w)\,p_{prior}, \qquad w = \frac{n}{n+k}

    Una tendencia de Linemate del tipo "8 de los ultimos 10 partidos" tiene n=10:
    su error estandar es ~0.13, de modo que tomarla como probabilidad puntual es
    un caso de libro de la **falacia del tamano muestral pequeno** (Tversky &
    Kahneman, 1971). El encogimiento hacia la linea neutralizada del mercado es la
    correccion conservadora: obliga a que el senal supere el ruido para generar EV.
    """
    if n < 0:
        raise ValueError("n debe ser no negativo.")
    w = n / (n + k)
    return float(np.clip(w * p_hat + (1.0 - w) * p_prior, 1e-6, 1 - 1e-6))


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Transformacion logit (log-odds)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    out = np.log(p / (1 - p))
    return float(out) if out.ndim == 0 else out


def inv_logit(x: float | np.ndarray) -> float | np.ndarray:
    """Funcion logistica inversa."""
    out = 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# 7. Contenedor de una seleccion evaluada                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Selection:
    """Una seleccion evaluada: precio de BetPlay + probabilidad estimada propia."""
    match_id: str
    market: str
    side: str
    line: float | None
    odds: float
    p_market_devig: float
    p_est: float
    trend_score: float
    method: str = "proportional"

    @property
    def ev(self) -> float:
        return expected_value(self.p_est, self.odds)

    @property
    def margin(self) -> float:
        """Discrepancia entre creencia propia y precio neutralizado del mercado."""
        return self.p_est - self.p_market_devig

    @property
    def kelly(self) -> float:
        return kelly_fraction(self.p_est, self.odds, fraction=0.25)
