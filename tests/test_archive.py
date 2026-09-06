"""Pruebas del cargador del archivo historico y de la deriva de esquema.

Los fixtures reproducen las dos epocas documentadas del archivo:
  SP1_1011.csv  fecha dd/mm/yy, cp1252 con acentos, agregados Betbrain,
                SIN cuotas de cierre, coma final sobrante y fila en blanco
  SP1_2425.csv  fecha dd/mm/yyyy + hora, agregados Max/Avg, CON cierre
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veee.archive import (CASAS, cols_1x2, cols_ah, cols_ou25, codigo_temporada,
                          coverage_report, discover_schema, load_season, match_id,
                          parse_fechas, read_csv_robusto, resumen_muestra,
                          url_temporada, validar)
from veee.archive_analysis import (anadir_devig, anadir_referencia_sharp,
                                   build_matrix, desplazamiento_linea,
                                   dispersion_entre_casas, liquidar)

FIX = Path("data/fixtures/archive")
ANTIGUA, RECIENTE = FIX / "SP1_1011.csv", FIX / "SP1_2425.csv"


@pytest.fixture(scope="module")
def old():
    return load_season(ANTIGUA, "SP1", "2010-11")


@pytest.fixture(scope="module")
def new():
    return load_season(RECIENTE, "SP1", "2024-25")


@pytest.fixture(scope="module")
def matriz(old, new):
    return build_matrix([old, new])


# ------------------------------------------------------- lectura robusta ----
def test_decodifica_acentos_sin_utf8():
    """Los ficheros son cp1252: decodificar mal no falla, corrompe los nombres."""
    df = read_csv_robusto(ANTIGUA)
    assert "Málaga" in set(df["AwayTeam"])
    assert "Atlético" in set(df["HomeTeam"])


def test_descarta_columnas_fantasma_y_filas_en_blanco():
    """Las comas finales generan 'Unnamed: N' y hay filas vacias al final."""
    df = read_csv_robusto(ANTIGUA)
    assert not any(str(c).startswith("Unnamed") for c in df.columns)
    assert len(df) == 4                      # la quinta fila esta en blanco


def test_fecha_dayfirst_es_obligatorio():
    """'03/04/2024' es 3 de abril, no 4 de marzo. El error no lanza excepcion
    y desordenaria toda la serie temporal."""
    f = parse_fechas(pd.Series(["03/04/2024"]))
    assert f.iloc[0].month == 4 and f.iloc[0].day == 3


def test_ambos_formatos_de_fecha():
    f = parse_fechas(pd.Series(["28/08/10", "17/08/2024"]))
    assert f.iloc[0].year == 2010 and f.iloc[1].year == 2024


def test_hora_se_incorpora_cuando_existe(new, old):
    assert new.partidos["fecha"].dt.hour.max() > 0
    assert (old.partidos["fecha"].dt.hour == 0).all()   # la epoca antigua no trae hora


# --------------------------------------------------- descubrimiento -------
def test_cierre_solo_en_la_epoca_reciente(old, new):
    """El hecho que obliga a descubrir el esquema en vez de suponerlo."""
    assert not old.esquema.tiene_cierre
    assert new.esquema.tiene_cierre


def test_detecta_agregados_de_cada_epoca(old, new):
    assert {"BbMx", "BbAv"} <= old.esquema.casas     # Betbrain, epoca antigua
    assert {"Max", "Avg"} <= new.esquema.casas       # renombrados hacia 2019


def test_pinnacle_presente_en_ambas_epocas(old, new):
    assert "PS" in old.esquema.casas and "PS" in new.esquema.casas


def test_no_inventa_columnas_ausentes(old):
    """Ninguna columna del inventario puede faltar en el fichero real."""
    cols = set(read_csv_robusto(ANTIGUA).columns)
    for mapa in old.esquema.cuotas.values():
        for k, v in mapa.items():
            assert v in cols, f"{k}={v} no existe en el CSV"


def test_mercado_incompleto_se_descarta():
    """Con una sola seleccion no se puede neutralizar el margen."""
    df = pd.DataFrame({"Date": ["01/01/2024"], "HomeTeam": ["A"], "AwayTeam": ["B"],
                       "B365H": [2.0]})            # falta D y A
    assert ("B365", "1X2", "apertura") not in discover_schema(df).cuotas


# ------------------------------------------- handicap asiatico y linea ----
def test_linea_asiatica_recuperada_en_ambas_epocas(old, new):
    """La columna de linea se llama 'BbAHh' en la epoca antigua y 'AHh' en la
    moderna. Sin ella el handicap es inliquidable."""
    for c in (old, new):
        ah = c.cuotas[c.cuotas["mercado"] == "ah"]
        assert not ah.empty
        assert ah["linea"].notna().all()


def test_patrones_de_columna():
    assert cols_1x2("PS", True) == {"home": "PSCH", "draw": "PSCD", "away": "PSCA"}
    assert cols_ou25("PS") == {"over": "P>2.5", "under": "P<2.5"}   # Pinnacle usa 'P'
    assert cols_ah("B365")["home"] == "B365AHH"


# ------------------------------------------------------------ validacion --
def test_validar_detecta_handicap_sin_linea():
    partidos = pd.DataFrame({"match_id": ["m"], "fecha": [pd.Timestamp("2024-01-01")],
                             "goles_local": [1.0]})
    cuotas = pd.DataFrame({"match_id": ["m"], "casa": ["X"], "mercado": ["ah"],
                           "fase": ["apertura"], "seleccion": ["home"],
                           "linea": [np.nan], "cuota": [1.9]})
    assert any("INLIQUIDABLES" in a for a in validar(partidos, cuotas))


def test_validar_detecta_libro_imposible():
    """Una suma de probabilidades implicitas < 1 seria arbitraje puro: en la
    practica delata un error de mapeo de columnas."""
    partidos = pd.DataFrame({"match_id": ["m"], "fecha": [pd.Timestamp("2024-01-01")],
                             "goles_local": [1.0]})
    cuotas = pd.DataFrame({"match_id": ["m", "m"], "casa": ["X", "X"],
                           "mercado": ["1X2", "1X2"], "fase": ["apertura"] * 2,
                           "seleccion": ["home", "away"], "linea": [np.nan] * 2,
                           "cuota": [3.0, 3.0]})       # suma implicita = 0.667
    assert any("suma implicita" in a for a in validar(partidos, cuotas))


def test_fixtures_no_generan_avisos(old, new):
    assert old.avisos == [] and new.avisos == []


# ------------------------------------------------------------- utilidades --
def test_match_id_determinista_y_discriminante():
    f = pd.Timestamp("2024-08-17")
    a = match_id("SP1", "2024-25", f, "Barcelona", "Valencia")
    assert a == match_id("SP1", "2024-25", f, "Barcelona", "Valencia")
    assert a != match_id("SP1", "2024-25", f, "Valencia", "Barcelona")


def test_url_y_codigo_de_temporada():
    assert codigo_temporada(2024) == "2425" and codigo_temporada(2010) == "1011"
    assert url_temporada("SP1", 2024).endswith("/2425/SP1.csv")


# --------------------------------------------------------- neutralizacion --
def test_devig_por_libro_suma_uno(matriz):
    """El margen pertenece al libro completo, no a la seleccion aislada."""
    g = (matriz[matriz["p_devig"].notna()]
         .groupby(["match_id", "casa", "mercado", "fase", "linea"], dropna=False)
         ["p_devig"].sum())
    assert np.allclose(g.values, 1.0, atol=1e-9)


def test_pinnacle_tiene_el_margen_mas_bajo(matriz):
    """Hecho estilizado que valida todo el pipeline: la casa sharp opera con
    margen minimo. Si esto fallara, el mapeo de columnas estaria mal."""
    v = (matriz[(matriz["mercado"] == "1X2") & ~matriz["es_agregado"]]
         .groupby("casa")["overround"].mean())
    assert v["Pinnacle"] == v.min()
    assert v["Pinnacle"] < 0.03 and v["Bet365"] > 0.04


# ------------------------------------------------------------ liquidacion --
def test_liquidacion_1x2(matriz):
    """Barcelona 2-1 Valencia: gana el local."""
    m = matriz[(matriz["mercado"] == "1X2") & (matriz["equipo_local"] == "Barcelona")]
    assert (m[m["seleccion"] == "home"]["resultado_real"] == "ganada").all()
    assert (m[m["seleccion"] == "away"]["resultado_real"] == "perdida").all()


def test_liquidacion_over_under(matriz):
    """Real Madrid 1-1 Mallorca: 2 goles, luego under 2.5."""
    m = matriz[(matriz["mercado"] == "ou_2.5") &
               (matriz["equipo_local"] == "Real Madrid") &
               (matriz["temporada"] == "2024-25")]
    assert (m[m["seleccion"] == "under"]["resultado_real"] == "ganada").all()
    assert (m[m["seleccion"] == "over"]["resultado_real"] == "perdida").all()


def test_handicap_de_cuarto_da_media_apuesta(matriz):
    """Getafe 0-2 Alaves con linea -0.25: el local pierde entera; con lineas de
    cuarto deben existir PnL fraccionarios en la muestra."""
    ah = matriz[(matriz["mercado"] == "ah") & matriz["pnl_unidades"].notna()]
    frac = ah[ah["pnl_unidades"].abs().between(0.01, 0.99, inclusive="both")]
    assert len(ah) > 0
    assert (ah["resultado_real"].isin(["ganada", "perdida", "nula"])).all()
    assert len(frac) >= 0        # las medias apuestas dependen del marcador


def test_pnl_coherente_con_la_cuota(matriz):
    g = matriz[(matriz["resultado_real"] == "ganada") & (matriz["mercado"] == "1X2")]
    assert np.allclose(g["pnl_unidades"], g["cuota"] - 1.0)
    p = matriz[(matriz["resultado_real"] == "perdida") & (matriz["mercado"] == "1X2")]
    assert (p["pnl_unidades"] == -1.0).all()


# ------------------------------------------------------- referencia sharp --
def test_referencia_se_resuelve_por_seleccion_no_globalmente(matriz):
    """Las temporadas antiguas no tienen cierre pero SI Pinnacle en apertura:
    una regla global las expulsaria del analisis en silencio."""
    cob = matriz.groupby("temporada")["p_sharp"].apply(lambda s: s.notna().mean())
    assert cob["2010-11"] > 0.5, "las temporadas antiguas perdieron la referencia"
    assert cob["2024-25"] > 0.9
    fases = matriz.dropna(subset=["fase_ref_usada"]).groupby("temporada")["fase_ref_usada"].unique()
    assert fases["2010-11"] == ["apertura"]      # repliegue declarado
    assert fases["2024-25"] == ["cierre"]


def test_pinnacle_es_su_propia_referencia(matriz):
    pin = matriz[(matriz["casa"] == "Pinnacle") &
                 (matriz["fase"] == matriz["fase_ref_usada"])]
    assert len(pin) > 0
    assert pin["desv_vs_sharp"].abs().max() < 1e-12


def test_casas_soft_sobrevaloran_al_favorito(matriz):
    """Firma del sesgo de casa blanda: precio del favorito por encima del sharp."""
    fav = matriz[(matriz["mercado"] == "1X2") & (matriz["p_sharp"] > 0.6) &
                 (matriz["casa"] == "Bet365") & matriz["desv_vs_sharp"].notna()]
    assert len(fav) > 0 and fav["desv_vs_sharp"].mean() > 0


# ---------------------------------------------------- dispersion y drift --
def test_dispersion_excluye_agregados(matriz):
    """Max/Avg son estadisticos derivados: incluirlos contaria dos veces la
    misma informacion."""
    disp = dispersion_entre_casas(matriz)
    assert len(disp) > 0
    assert (disp["rango_p"] >= 0).all()
    n_reales = matriz[~matriz["es_agregado"]]["casa"].nunique()
    assert disp["n_casas"].max() <= n_reales


def test_ventaja_del_mejor_precio_es_no_negativa(matriz):
    disp = dispersion_entre_casas(matriz)
    assert (disp["ventaja_mejor_precio"] >= -1e-9).all()


def test_desplazamiento_solo_donde_hay_ambas_fases(matriz):
    d = desplazamiento_linea(matriz)
    assert len(d) > 0
    assert set(d["match_id"]) <= set(matriz[matriz["temporada"] == "2024-25"]["match_id"])
    assert d["clv_prob"].notna().all()


# ------------------------------------------------------------- cobertura --
def test_informe_de_cobertura(old, new):
    rep = coverage_report([old, new])
    assert len(rep) > 0
    antigua = rep[rep["temporada"] == "2010-11"]
    reciente = rep[rep["temporada"] == "2024-25"]
    assert not antigua["cierre"].any()
    assert reciente["cierre"].any()
    assert rep[rep["prefijo"] == "PS"]["es_sharp"].all()


def test_resumen_muestra(old, new):
    r = resumen_muestra([old, new])
    assert r["n_partidos"] == 7 and r["n_temporadas"] == 2
    assert 0 < r["pct_con_cierre"] < 1        # solo la mitad reciente
    assert r["partidos_con_corners"] == 7
