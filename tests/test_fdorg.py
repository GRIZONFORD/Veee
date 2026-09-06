"""Pruebas del conector football-data.org y del emparejamiento entre fuentes."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from veee.crosswalk import (emparejar_equipos, emparejar_partidos,
                            informe_emparejamiento, normalizar_nombre)
from veee.fdorg import (A_DIV_ARCHIVO, ENV_TOKEN, FootballDataOrg,
                        FootballDataOrgError, _mascara, _trae_cuotas,
                        normalizar_partidos, tabla_equipos)

FIX = Path("data/fixtures/fdorg/PD_2024_matches.json")


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def partidos(payload):
    return normalizar_partidos(payload)


# ----------------------------------------------------------- credencial ----
def test_token_nunca_aparece_completo():
    """Un token en un registro o en el repositorio queda expuesto para siempre."""
    m = _mascara("101d61af56784a4ca3f3061a120305c4")
    assert m == "...05c4"
    assert "101d61af" not in m


def test_falta_de_token_da_instruccion_accionable(monkeypatch):
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    with pytest.raises(FootballDataOrgError, match=ENV_TOKEN):
        FootballDataOrg(token="")


def test_token_se_lee_del_entorno(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "x" * 32)
    c = FootballDataOrg()
    assert c.token == "x" * 32


# ------------------------------------------------------ normalizacion -----
def test_partidos_normalizados(partidos):
    assert len(partidos) == 4
    assert set(partidos["competicion"]) == {"PD"}
    assert set(partidos["div"]) == {"SP1"}          # equivalencia con el archivo
    assert partidos["temporada"].iloc[0] == "2024-25"


def test_aplazado_no_se_lee_como_cero_a_cero(partidos):
    """Un POSTPONED con marcador nulo se colaria como empate 0-0 sin filtrar."""
    apl = partidos[partidos["estado"] == "POSTPONED"]
    assert len(apl) == 1
    assert apl["goles_local"].isna().all()
    assert apl["resultado_ft"].isna().all()


def test_resultados_traducidos(partidos):
    fin = partidos[partidos["estado"] == "FINISHED"].set_index("fd_match_id")
    assert fin.loc[497567, "resultado_ft"] == "H"
    assert fin.loc[497568, "resultado_ft"] == "D"
    assert fin.loc[497569, "resultado_ft"] == "A"
    assert fin.loc[497569, "margen_goles"] == -2
    assert fin.loc[497567, "goles_totales"] == 3


def test_identificadores_estables(partidos):
    """El id numerico y el tla son el ancla del emparejamiento entre fuentes."""
    eq = tabla_equipos(partidos)
    assert len(eq) == 8
    assert eq["team_id"].is_unique
    assert set(eq[eq["nombre"] == "FC Barcelona"]["tla"]) == {"FCB"}


def test_plan_gratuito_no_trae_cuotas(payload):
    """football-data.org NO sustituye a Football-Data.co.uk: no sirve precios."""
    assert _trae_cuotas(payload) is False


def test_hora_utc_preservada(partidos):
    """Sin la hora del pitido no se puede anclar la escalera de captura."""
    b = partidos[partidos["fd_match_id"] == 497567].iloc[0]
    assert b["fecha"].hour == 17


# -------------------------------------------------------- emparejamiento --
def test_normalizacion_de_nombres():
    assert normalizar_nombre("FC Barcelona") == "barcelona"
    assert normalizar_nombre("Deportivo Alavés") == "deportivo alaves"
    assert normalizar_nombre("Athletic Club") == "athletic"
    assert normalizar_nombre("Ath Bilbao") == "athletic"     # via alias


def test_cascada_empareja_variantes_reales():
    fdorg = ["FC Barcelona", "Real Madrid CF", "Club Atlético de Madrid",
             "Athletic Club", "Deportivo Alavés", "Real Sociedad de Fútbol"]
    archivo = ["Barcelona", "Real Madrid", "Ath Madrid", "Ath Bilbao",
               "Alaves", "Sociedad"]
    df = emparejar_equipos(fdorg, archivo)
    assert not df["requiere_revision"].any()
    assert df.set_index("origen").loc["Athletic Club", "destino"] == "Ath Bilbao"
    assert df.set_index("origen").loc["Deportivo Alavés", "destino"] == "Alaves"


def test_ambiguedad_espanyol_barcelona_no_se_resuelve_por_conjetura():
    """TRAMPA CENTRAL: 'Barcelona' es subconjunto de 'FC Barcelona' y de
    'RCD Espanyol de Barcelona'. Emparejar por conjetura confundiria dos clubes
    rivales de la misma ciudad y contaminaria el estudio en silencio."""
    df = emparejar_equipos(["RCD Espanyol de Barcelona"], ["Barcelona", "Espanol"])
    fila = df.iloc[0]
    assert fila["metodo"] == "ambiguo"
    assert fila["destino"] is None
    assert fila["requiere_revision"]
    assert "Barcelona" in fila["nota"] and "Espanol" in fila["nota"]


def test_difuso_siempre_requiere_revision():
    df = emparejar_equipos(["Wolverhamptn Wandrers"], ["Wolverhampton Wanderers"])
    assert df.iloc[0]["metodo"] in ("difuso", "sin_emparejar")
    assert df.iloc[0]["requiere_revision"]


def test_sin_emparejar_se_declara():
    df = emparejar_equipos(["Equipo Inexistente"], ["Barcelona", "Valencia"])
    assert df.iloc[0]["metodo"] == "sin_emparejar"
    assert df.iloc[0]["destino"] is None


def test_informe_de_emparejamiento():
    df = emparejar_equipos(["FC Barcelona", "Equipo Raro"], ["Barcelona"])
    inf = informe_emparejamiento(df)
    assert inf["n_equipos"] == 2 and inf["automaticos"] == 1
    assert inf["cobertura_automatica"] == 0.5
    assert len(inf["a_revisar"]) == 1


def test_emparejar_partidos_tolera_desfase_de_un_dia():
    """Un partido a las 21:00 en Madrid cae en otro dia en UTC segun la fuente."""
    a = pd.DataFrame({"fecha": ["2024-08-18T23:30:00Z"],
                      "equipo_local": ["Real Madrid CF"],
                      "equipo_visitante": ["RCD Mallorca"]})
    b = pd.DataFrame({"fecha": ["2024-08-19"], "equipo_local": ["Real Madrid"],
                      "equipo_visitante": ["Mallorca"]})
    f = emparejar_partidos(a, b)
    assert len(f) == 1 and f.iloc[0]["dias_diferencia"] <= 1


def test_emparejar_partidos_vacio_no_revienta():
    a = pd.DataFrame({"fecha": ["2024-08-18"], "equipo_local": ["Aaa"],
                      "equipo_visitante": ["Bbb"]})
    b = pd.DataFrame({"fecha": ["2024-08-18"], "equipo_local": ["Ccc"],
                      "equipo_visitante": ["Ddd"]})
    assert emparejar_partidos(a, b).empty


def test_cruce_real_fdorg_contra_archivo(partidos):
    """Prueba de integracion: los partidos del payload cruzan con el fixture del
    archivo CSV, que usa nombres distintos."""
    from veee.archive import load_season
    arch = load_season("data/fixtures/archive/SP1_2425.csv", "SP1", "2024-25")
    mapa = {r["origen"]: r["destino"] for _, r in
            emparejar_equipos(
                sorted(set(partidos["equipo_local"]) | set(partidos["equipo_visitante"])),
                sorted(set(arch.partidos["equipo_local"]) |
                       set(arch.partidos["equipo_visitante"]))).iterrows()
            if r["destino"]}
    fus = emparejar_partidos(partidos, arch.partidos,
                             mapa_equipos={k: normalizar_nombre(v) for k, v in mapa.items()})
    assert len(fus) >= 3, f"solo cruzaron {len(fus)} partidos"
