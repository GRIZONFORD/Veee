"""Pruebas del conector ClubElo: parseo, busqueda a fecha (as-of), crosswalk de
nombres y el logit ordenado que convierte diferencia de Elo en P(1X2).

La validacion central del modelo (`test_ajustar_modelo_recupera_parametros_...`)
sigue el mismo patron que `tests/test_econometrics.py`: generar datos con un
proceso conocido y comprobar que el ajuste recupera los parametros verdaderos,
porque no hay suficientes partidos reales en los fixtures del repo (n=7) para
validar la maquinaria estadistica de otro modo.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veee import clubelo as ce
from veee.archive import load_season
from veee.archive_analysis import build_matrix

FIX = Path("data/fixtures/clubelo")
ARCHIVO_ANTIGUA = Path("data/fixtures/archive/SP1_1011.csv")
ARCHIVO_RECIENTE = Path("data/fixtures/archive/SP1_2425.csv")


@pytest.fixture(scope="module")
def snapshot():
    return ce.cargar_snapshot(FIX / "snapshots/2024-08-17.csv")


@pytest.fixture(scope="module")
def historiales():
    out = {}
    for p in (FIX / "historial").glob("*.csv"):
        h = ce.cargar_historial(p)
        out[h["club"].iloc[0]] = h
    return out


@pytest.fixture(scope="module")
def matriz_archivo():
    old = load_season(ARCHIVO_ANTIGUA, "SP1", "2010-11")
    new = load_season(ARCHIVO_RECIENTE, "SP1", "2024-25")
    return build_matrix([old, new])


# --------------------------------------------------------------- parseo CSV --
def test_parse_valida_columnas_esperadas():
    csv = b"Rank,Club,Country,Level,Elo,From,To\n1,Barcelona,ESP,1,1980,2015-07-01,2020-06-30\n"
    df = ce.parse_csv_robusto(csv)
    assert list(df.columns) == ["rank", "club", "country", "level", "elo", "from", "to"]
    assert df["elo"].iloc[0] == 1980.0


def test_parse_rechaza_esquema_desconocido():
    """Si ClubElo cambiara sus columnas, debe fallar alto y claro, no producir
    NaN silenciosos mas adelante."""
    csv = b"Ranking,Team,Nation,Tier,Rating,Start,End\n1,Barcelona,ESP,1,1980,2015-07-01,2020-06-30\n"
    with pytest.raises(ValueError, match="faltan columnas"):
        ce.parse_csv_robusto(csv)


def test_parse_descarta_intervalos_invertidos():
    csv = (b"Rank,Club,Country,Level,Elo,From,To\n"
          b"1,Bueno,ESP,1,1900,2020-01-01,2020-12-31\n"
          b"2,Malo,ESP,1,1800,2021-01-01,2019-01-01\n")   # From > To
    df = ce.parse_csv_robusto(csv)
    assert list(df["club"]) == ["Bueno"]


def test_parse_descarta_fechas_no_interpretables():
    csv = (b"Rank,Club,Country,Level,Elo,From,To\n"
          b"1,Bueno,ESP,1,1900,2020-01-01,2020-12-31\n"
          b"2,Malo,ESP,1,1800,no-es-fecha,2020-12-31\n")
    df = ce.parse_csv_robusto(csv)
    assert list(df["club"]) == ["Bueno"]


# ------------------------------------------------------------ busqueda as-of --
def _historial(tramos):
    filas = [{"rank": 1, "club": "X", "country": "ESP", "level": 1, "elo": elo,
             "from": pd.Timestamp(a), "to": pd.Timestamp(b)} for a, b, elo in tramos]
    return pd.DataFrame(filas)


def test_elo_as_of_dentro_del_intervalo():
    h = _historial([("2020-01-01", "2020-12-31", 1800), ("2021-01-01", "2021-12-31", 1850)])
    elo, estado = ce.elo_as_of(h, pd.Timestamp("2020-06-15"))
    assert elo == 1800 and estado == "ok"
    elo, estado = ce.elo_as_of(h, pd.Timestamp("2021-06-15"))
    assert elo == 1850 and estado == "ok"


def test_elo_as_of_no_extrapola_fuera_de_rango():
    """Una fecha sin cobertura es NaN, nunca una aproximacion silenciosa."""
    h = _historial([("2020-01-01", "2020-12-31", 1800)])
    elo, estado = ce.elo_as_of(h, pd.Timestamp("2015-01-01"))
    assert np.isnan(elo) and estado == "anterior_al_primer_registro"
    elo, estado = ce.elo_as_of(h, pd.Timestamp("2025-01-01"))
    assert np.isnan(elo) and estado == "posterior_al_ultimo_registro"


def test_elo_as_of_detecta_hueco_entre_intervalos():
    h = _historial([("2020-01-01", "2020-06-30", 1800), ("2020-08-01", "2020-12-31", 1820)])
    elo, estado = ce.elo_as_of(h, pd.Timestamp("2020-07-15"))
    assert np.isnan(elo) and estado == "hueco_entre_intervalos"


def test_elo_as_of_historial_vacio():
    elo, estado = ce.elo_as_of(pd.DataFrame(columns=["from", "to", "elo"]),
                               pd.Timestamp("2020-01-01"))
    assert np.isnan(elo) and estado == "sin_datos"


# -------------------------------------------------------------- crosswalk ----
def test_mapa_desde_snapshot_resuelve_grafias_distintas(snapshot):
    """ClubElo usa su propia grafia: 'Atletico Madrid' frente a 'Atlético' del
    archivo, 'Malaga' sin tilde frente a 'Málaga'."""
    cw = ce.mapa_desde_snapshot(snapshot, ["Atlético", "Málaga", "Barcelona"])
    m = cw.set_index("origen")
    assert m.loc["Atlético", "destino"] == "Atletico Madrid"
    assert m.loc["Málaga", "destino"] == "Malaga"
    assert not cw["requiere_revision"].any()


def test_mapa_desde_snapshot_filtra_por_pais(snapshot):
    """El snapshot fixture incluye un club ingles a proposito: sin el filtro de
    pais, podria emparejarse por error con un nombre espanol parecido."""
    assert "Manchester City" in set(snapshot["club"])
    cw_esp = ce.mapa_desde_snapshot(snapshot, ["Barcelona"], pais="ESP")
    assert cw_esp.iloc[0]["destino"] == "Barcelona"


# -------------------------------------------------- enriquecimiento del archivo --
def test_enriquecer_con_elo_usa_el_tramo_correcto_por_fecha(matriz_archivo, snapshot,
                                                             historiales):
    """Barcelona y Real Madrid aparecen en 2010 Y en 2024 en los fixtures: el
    join as-of debe distinguir el tramo de cada epoca, no tomar un unico valor
    fijo por equipo."""
    nombres = sorted(set(matriz_archivo.equipo_local) | set(matriz_archivo.equipo_visitante))
    cw = ce.mapa_desde_snapshot(snapshot, nombres)
    mapa = {r.origen: r.destino for r in cw.itertuples() if r.destino}
    partidos = (matriz_archivo[["match_id", "fecha", "equipo_local", "equipo_visitante",
                                "resultado_ft"]].drop_duplicates("match_id"))
    enr = ce.enriquecer_con_elo(partidos, historiales, mapa)

    assert enr["elo_cobertura"].all()
    # `fecha` llega como objeto date de Python (asi la deja archive_analysis),
    # no como Timestamp: enriquecer_con_elo la reconvierte SOLO internamente
    # para el calculo, sin normalizar la columna de salida.
    anio = pd.to_datetime(enr["fecha"]).dt.year
    bcn_2010 = enr[(enr.equipo_local == "Barcelona") & (anio == 2010)]
    bcn_2024 = enr[(enr.equipo_local == "Barcelona") & (anio == 2024)]
    assert bcn_2010.iloc[0]["elo_local"] != bcn_2024.iloc[0]["elo_local"]


def test_enriquecer_con_elo_marca_falta_de_mapa():
    partidos = pd.DataFrame({"match_id": ["m1"], "fecha": [pd.Timestamp("2020-01-01")],
                             "equipo_local": ["Equipo Fantasma"],
                             "equipo_visitante": ["Barcelona"]})
    hist = {"Barcelona": _historial([("2019-01-01", "2021-01-01", 1900)])}
    enr = ce.enriquecer_con_elo(partidos, hist, {"Barcelona": "Barcelona"})
    assert enr.iloc[0]["elo_local_estado"] == "sin_mapa_clubelo"
    assert not enr.iloc[0]["elo_cobertura"]


# --------------------------------------------------- logit ordenado (p_elo) --
def _dgp_ordenado(n, beta, kappa, seed):
    """Genera partidos sinteticos con un proceso de logit ordenado conocido."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 220, n)
    eta = beta * elo_diff
    p_away = 1 / (1 + np.exp(-(kappa[0] - eta)))
    p_away_o_draw = 1 / (1 + np.exp(-(kappa[1] - eta)))
    u = rng.random(n)
    y = np.where(u < p_away, "A", np.where(u < p_away_o_draw, "D", "H"))
    return pd.DataFrame({"match_id": range(n), "fecha": pd.Timestamp("2024-01-01"),
                         "elo_diff": elo_diff, "elo_cobertura": True, "resultado_ft": y})


def test_guardrail_de_muestra_minima():
    df = _dgp_ordenado(7, 0.0045, [-0.5, 0.5], seed=1)
    with pytest.raises(ValueError, match="insuficiente"):
        ce.ajustar_modelo_elo(df)


def test_ajustar_modelo_recupera_parametros_conocidos():
    """Validacion central: sin partidos reales suficientes en los fixtures,
    se comprueba la maquinaria estadistica contra un proceso generador con
    parametros conocidos, igual que test_econometrics.py hace para el resto
    del estudio."""
    beta_real, kappa_real = 0.0045, [-0.55, 0.55]
    df = _dgp_ordenado(3000, beta_real, kappa_real, seed=42)
    modelo = ce.ajustar_modelo_elo(df)

    beta_hat = modelo.ajuste.params["elo_diff"]
    assert beta_hat == pytest.approx(beta_real, rel=0.25)

    ev = ce.evaluar_modelo_elo(modelo, df)
    k0, k1 = ev["puntos_de_corte"]
    assert k0 == pytest.approx(kappa_real[0], abs=0.15)
    assert k1 == pytest.approx(kappa_real[1], abs=0.15)


def test_modelo_bate_a_la_prediccion_marginal():
    """Con senal real en el DGP, el Brier del modelo debe ser mejor (menor) que
    predecir siempre la frecuencia marginal observada."""
    df = _dgp_ordenado(2000, 0.006, [-0.4, 0.4], seed=5)
    modelo = ce.ajustar_modelo_elo(df)
    ev = ce.evaluar_modelo_elo(modelo, df)
    assert ev["brier_elo"] < ev["brier_base_marginal"]
    assert ev["mejora_vs_marginal"] > 0


def test_predecir_direccion_correcta():
    """A igualdad de Elo, las tres probabilidades deben sumar 1; con el local
    mucho mas fuerte, P(home) debe ser la mayor."""
    df = _dgp_ordenado(2000, 0.006, [-0.4, 0.4], seed=9)
    modelo = ce.ajustar_modelo_elo(df)
    pred = modelo.predecir([0, 300, -300])
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    assert pred.iloc[1]["p_home"] > pred.iloc[1]["p_away"]      # local +300 Elo
    assert pred.iloc[2]["p_away"] > pred.iloc[2]["p_home"]      # local -300 Elo


def test_comparar_con_mercado_alineacion_correcta():
    """La comparacion debe usar la fila de p_elo correspondiente a la
    seleccion (home/draw/away) de cada apuesta, no una columna fija."""
    df = _dgp_ordenado(1500, 0.006, [-0.4, 0.4], seed=3)
    modelo = ce.ajustar_modelo_elo(df)
    pred = modelo.predecir(df["elo_diff"].values)

    resultado_ft = df["resultado_ft"].reset_index(drop=True)
    seleccion = pd.Series(["home"] * len(df))       # todas apuestan al local
    p_mercado = pd.Series(np.full(len(df), 0.5))    # precio de mercado ficticio

    comp = ce.comparar_con_mercado(pred, resultado_ft, p_mercado, seleccion)
    # Con seleccion constante "home", el brier manual coincide con el propio p_home.
    y = (resultado_ft == "H").astype(float).values
    esperado = float(np.mean((pred["p_home"].values - y) ** 2))
    assert comp["brier_elo"] == pytest.approx(esperado)
    assert comp["n"] == len(df)


# ------------------------------------------------------------------- humo ---
def test_probar_conexion_no_lanza_y_devuelve_tupla(tmp_path):
    ok, detalle = ce.probar_conexion(cache_dir=tmp_path, timeout=3, max_retries=1)
    assert isinstance(ok, bool) and isinstance(detalle, str)
