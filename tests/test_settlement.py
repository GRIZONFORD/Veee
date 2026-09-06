"""Pruebas de las reglas de liquidacion, con enfasis en handicaps asiaticos."""
import pytest

from veee.settlement import liquidar_ou, pnl_ah, pnl_binario


def test_ou_basico():
    assert liquidar_ou(11, 9.5, "over") == "ganada"
    assert liquidar_ou(11, 9.5, "under") == "perdida"
    assert liquidar_ou(8, 9.5, "over") == "perdida"


def test_ou_push_en_linea_entera():
    """Con linea entera y valor igual, la apuesta es nula (devolucion)."""
    assert liquidar_ou(10, 10.0, "over") == "nula"
    assert liquidar_ou(10, 10.0, "under") == "nula"


def test_pnl_binario():
    assert pnl_binario("ganada", 2.10) == pytest.approx(1.10)
    assert pnl_binario("perdida", 2.10) == -1.0
    assert pnl_binario("nula", 2.10) == 0.0


def test_ah_linea_entera_push():
    """Handicap 0.0 con empate: devolucion integra."""
    assert pnl_ah(0.0, 0.0, 1.90) == ("nula", 0.0)


def test_ah_media_derrota_en_cuarto():
    """-0.25 con empate: se pierde la mitad del stake."""
    etiqueta, pnl = pnl_ah(0.0, -0.25, 1.90)
    assert etiqueta == "perdida" and pnl == pytest.approx(-0.5)


def test_ah_media_victoria_en_cuarto():
    """+0.25 con empate: se gana la mitad de la ganancia."""
    etiqueta, pnl = pnl_ah(0.0, 0.25, 1.90)
    assert etiqueta == "ganada" and pnl == pytest.approx(0.45)


def test_ah_cuarto_negativo_victoria_por_uno():
    """-0.75 ganando por 1: media victoria."""
    etiqueta, pnl = pnl_ah(1.0, -0.75, 1.90)
    assert etiqueta == "ganada" and pnl == pytest.approx(0.45)


def test_ah_victoria_completa():
    assert pnl_ah(2.0, -0.5, 1.90)[1] == pytest.approx(0.90)
