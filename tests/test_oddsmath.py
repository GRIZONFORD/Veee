"""Pruebas de la aritmetica de cuotas y de la eliminacion del margen."""
import numpy as np
import pytest

from veee import oddsmath as om


@pytest.mark.parametrize("odds", [[2.10, 3.40, 3.60], [1.92, 1.88], [1.45, 4.80, 6.50]])
@pytest.mark.parametrize("metodo", ["proportional", "additive", "power", "shin"])
def test_devig_suma_uno(odds, metodo):
    """Toda neutralizacion debe producir una distribucion de probabilidad valida."""
    p = om.devig(odds, metodo)
    assert np.isclose(p.sum(), 1.0, atol=1e-9)
    assert np.all(p > 0) and np.all(p < 1)


@pytest.mark.parametrize("metodo", ["proportional", "additive", "power", "shin"])
def test_devig_preserva_orden(metodo):
    """La neutralizacion no puede alterar el orden de las selecciones."""
    odds = [1.50, 4.00, 7.00]
    p = om.devig(odds, metodo)
    assert p[0] > p[1] > p[2]


def test_devig_sin_margen_es_identidad():
    """Con un libro justo (suma 1), toda neutralizacion devuelve las mismas probabilidades."""
    p_verdad = np.array([0.5, 0.3, 0.2])
    odds = 1.0 / p_verdad
    for m in ["proportional", "additive", "power", "shin"]:
        assert np.allclose(om.devig(odds, m), p_verdad, atol=1e-6)


def test_power_castiga_longshots_mas_que_proporcional():
    """El metodo de potencia corrige el sesgo favorito-longshot del proporcional."""
    odds = [1.30, 5.50, 12.00]
    p_prop, p_pow = om.devig(odds, "proportional"), om.devig(odds, "power")
    assert p_pow[-1] < p_prop[-1]        # menos probabilidad al longshot
    assert p_pow[0] > p_prop[0]          # mas al favorito


def test_overround_positivo_y_coherente():
    odds = [1.92, 1.88]
    assert om.overround(odds) == pytest.approx(om.booksum(odds) - 1.0)
    assert om.overround(odds) > 0


def test_ev_cero_en_cuota_justa():
    """Si la cuota es exactamente 1/p, el valor esperado es nulo."""
    assert om.expected_value(0.40, 1 / 0.40) == pytest.approx(0.0, abs=1e-12)


def test_ev_signo():
    assert om.expected_value(0.55, 2.00) > 0
    assert om.expected_value(0.45, 2.00) < 0


def test_devig_no_genera_ev_en_su_propio_libro():
    """Punto de identificacion: apostar al precio neutralizado de la MISMA casa
    produce EV nulo por construccion. El filtro +EV exige informacion externa."""
    odds = [2.10, 3.40, 3.60]
    p = om.devig(odds, "proportional")
    evs = [om.expected_value(pi, ci) for pi, ci in zip(p, odds)]
    assert np.average(evs, weights=p) == pytest.approx(-om.overround(odds) / om.booksum(odds),
                                                       abs=1e-9)


def test_breakeven_prob():
    assert om.breakeven_prob(2.50) == pytest.approx(0.40)


def test_kelly_acotado_y_no_negativo():
    assert om.kelly_fraction(0.30, 2.00) == 0.0        # EV negativo -> no apostar
    assert 0 < om.kelly_fraction(0.60, 2.00, cap=0.05) <= 0.05


def test_clv_signos():
    assert om.clv_odds_ratio(2.10, 1.95) > 0           # se tomo mejor precio que el cierre
    assert om.clv_odds_ratio(1.85, 2.00) < 0


def test_shrinkage_converge_al_prior_y_a_la_muestra():
    """Con n pequeno domina el prior; con n grande domina la frecuencia observada."""
    assert om.shrink_to_prior(0.9, 1, 0.5, k=10) < 0.55
    assert om.shrink_to_prior(0.9, 10_000, 0.5, k=10) > 0.89


def test_negbin_engrosa_las_colas():
    """La sobredispersion engrosa las colas: mas probabilidad de superar una linea alta.

    Consecuencia economica: modelar corners con Poisson cuando hay sobredispersion
    INFRAVALORA la probabilidad de los *overs* de linea alta y por tanto genera
    senales +EV espurias en el lado contrario.
    """
    media, linea_alta, linea_baja = 10.0, 13.5, 6.5
    assert om.negbin_over_prob(media, 16.0, linea_alta) > om.poisson_over_prob(media, linea_alta)
    # Simetricamente, menos masa en el centro-bajo de la distribucion.
    assert om.negbin_over_prob(media, 16.0, linea_baja) < om.poisson_over_prob(media, linea_baja)


def test_negbin_converge_a_poisson_sin_sobredispersion():
    """Si var <= media no hay sobredispersion y el modelo revierte a Poisson."""
    assert om.negbin_over_prob(10.0, 9.0, 11.5) == om.poisson_over_prob(10.0, 11.5)


def test_cuotas_invalidas_rechazadas():
    for mala in ([0.98, 2.0], [2.0, np.nan], [2.0, 5000.0]):
        with pytest.raises(ValueError):
            om.devig(mala)


def test_logit_inverso():
    for p in (0.01, 0.5, 0.99):
        assert om.inv_logit(om.logit(p)) == pytest.approx(p, abs=1e-9)
