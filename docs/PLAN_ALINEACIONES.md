# Plan de Acción — Predicción Pre-Partido desde la Alineación Titular

> Investigación en Kaggle y GitHub (vía WebSearch/WebFetch, con verificación de
> primera mano del código fuente de `soccerdata` donde la documentación no
> bastaba) para diseñar una fuente de probabilidad propia (`p_est`) basada en la
> alineación inicial y métricas de equipo/jugador, comparable después contra el
> resultado real. Este documento es el plan; no se ha escrito código todavía.

---

## 0. Cómo encaja esto con lo ya construido

Esto **no es un proyecto nuevo**: es una tercera fuente de `p_est`, paralela a
`p_shrunk` (tendencias de Linemate) y `p_count` (modelo Poisson/NB) que ya
combina `model.p_blend`. La misma lógica de identificación que justificó usar
Polymarket u otra casa aplica aquí: una probabilidad derivada de la alineación y
la fuerza de los jugadores es un conjunto de información **genuinamente
independiente** del precio de la casa.

Dos anclas ya existen en el código y no hay que rediseñarlas:

- `capture.VENTANAS_H` ya incluye **T-1h**, que es exactamente la ventana en la
  que las alineaciones titulares se confirman (verificado por tres fuentes
  independientes en la investigación, ver §1.1). Solo hace falta un evaluador
  que dispare en ese hito, no una escalera nueva.
- `model.p_blend(p_a, p_b, w_a, w_b)` combina hoy dos fuentes. Es un pool
  logarítmico (`logit(p̄) = (w_a·logit(p_a) + w_b·logit(p_b))/(w_a+w_b)`), que se
  generaliza a N fuentes sumando términos — es un refactor pequeño, no una
  reescritura.

---

## 1. Qué se investigó y qué se encontró

### 1.1 Fuentes de alineación pre-partido (el dato que dispara todo)

| Fuente | Qué ofrece | Ventana temporal | Riesgo legal |
|---|---|---|---|
| [TheStatsAPI](https://www.thestatsapi.com/football/lineups) | XI confirmado, formación, posición por jugador | Confirma que las alineaciones se publican **~T-60min** | Comercial, sin verificar |
| [SofaScore](https://github.com/probberechts/soccerdata) (vía `soccerdata`/`ScraperFC`) | Alineaciones y formaciones en vivo | En vivo | **Su [ToS](https://torneo.sofascore.com/terms-of-service) prohíbe explícitamente scraping, robots, crawling y automatización** |
| [Flashscore](https://github.com/msarnacki/flashscore-scraper) | Alineaciones, cuotas, incidencias | En vivo | ToS no verificado; mismo patrón de riesgo |
| [Sky Sports lineup scraper](https://github.com/jchadwick92/Skysports-lineup-scraper) | Envía email al publicarse la alineación | Dispara ~T-60min | Sitio no pensado para consumo automatizado |
| **FBref** `read_lineup()` (vía `soccerdata`) | Alineación con minutos jugados | **Solo POST-partido** (ver hallazgo abajo) | — |
| [StatsBomb open-data](https://github.com/statsbomb/open-data) | JSON de alineaciones + eventos + tácticas | Histórico | **Abierto, licencia libre** — pero cobertura curada por competición |

**Hallazgo obtenido leyendo el código fuente, no solo la documentación:** instalé
`soccerdata` e inspeccioné `fbref.py` directamente. Su `read_lineup()` lee la
página de *match report* de FBref, que **solo existe una vez jugado el
partido** (`df_schedule[...~df_schedule.match_report.isnull()]`). No sirve como
disparador pre-partido — únicamente como archivo retrospectivo para construir
features de entrenamiento a partir de partidos ya jugados. Confundir esto
habría sido exactamente el tipo de fuga de datos que se discute en §3.

### 1.2 Fuentes de fuerza de equipo/jugador (para features, no como disparador)

| Fuente | Qué ofrece | Fricción técnica/legal |
|---|---|---|
| [ClubElo](http://clubelo.com/System) | Un valor Elo por club y fecha, histórico desde 1939, vinculado directamente a probabilidad de victoria | **Verificado en el código**: `class ClubElo(BaseRequestsReader)` — API HTTP simple, sin navegador. La opción más limpia |
| [FBref](https://soccerdata.readthedocs.io/en/latest/datasources/FBref.html) (vía `soccerdata`) | Stats avanzadas por equipo/jugador: `shooting`, `keeper`, `misc`, `passing`, `possession` | **Verificado**: `class FBref(BaseSeleniumReader)` — exige un navegador real (Selenium), indicio de que el sitio está endurecido contra scraping simple |
| [Understat](https://github.com/gabo-01/understatAPI) | xG por partido y jugador, 6 ligas europeas desde 2014/15 | — |
| SoFIFA / datasets FIFA en Kaggle | Rating de jugador vía videojuego, 100+ atributos | Se actualiza por edición/parche, **no por jornada** — cuidado con la cadencia si se usa como "fuerza actual" |
| [withqwerty/availability-data](https://github.com/withqwerty/availability-data) | Lesiones y sanciones por club, `injury_burden`, `starting_matchdays`, `squad_players` | **Verificado leyendo el repo directamente**: 6 ligas incl. LaLiga, 2015/16–2025/26, **actualización semanal automatizada**, datos de Transfermarkt "con fines de investigación", publicado ya como CSV — no exige scrapear Transfermarkt uno mismo |

### 1.3 Datasets de Kaggle (nombrados)

| Dataset | Aporta |
|---|---|
| [adamgbor/club-football-match-data-2000-2025](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) | 42 ligas incl. LaLiga, explícitamente orientado a predicción pre-partido y en vivo, con cuotas |
| [enricocattaneo/data-football-match-prediction](https://www.kaggle.com/datasets/enricocattaneo/data-football-match-prediction) | Top-5 ligas 2016/17–2021/22, **incluye jugadores de la alineación titular por partido** (poco común) |
| [saurabhshahane/statsbomb-football-data](https://www.kaggle.com/datasets/saurabhshahane/statsbomb-football-data) | StatsBomb empaquetado para Kaggle |
| [mominullptr/FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset) | **65 features pre-calculadas** por partido: Elo, ranking FIFA, valor de plantilla, forma rolling (goles/xG/tiros/córners/posesión a 5 partidos), días de descanso, altitud del estadio. Es de Mundial, no de liga, pero el catálogo de familias de variables es directamente trasladable |
| [Kaggle: football-match-probability-prediction](https://www.kaggle.com/competitions/football-match-probability-prediction) | Competición con benchmark público — referencia de métrica de evaluación (log-loss) |

### 1.4 El hallazgo metodológico más relevante

**"A leakage-aware workflow for pre-match forecasting of secondary football
markets: A LaLiga case study"** (revista *Array*, ScienceDirect, 2026). No pude
acceder al texto completo — `sciencedirect.com` está bloqueado por la política
de red de este entorno igual que el resto de dominios externos — pero el
resumen ya es directamente accionable:

- **Mismo caso exacto**: LaLiga, 760 partidos fuera de muestra, feb-2024 a
  mar-2026.
- **Mismos mercados que ya cubre la Vía A del repo**: tiros, tiros a puerta,
  córners, tarjetas, faltas.
- Metodología: **resolución determinista de nombres de equipo** (exactamente lo
  que hace `crosswalk.py`), reconciliación de datos de partido/jugador,
  **features rolling "temporalmente seguras"** (calculadas solo con partidos
  estrictamente anteriores), **"agregación consciente de la alineación"**
  (agregar solo a los jugadores que efectivamente juegan, no a toda la
  plantilla), y evaluación de ventana expansiva — el mismo patrón de partición
  calibración/validación ya usado en `econometrics.py`.
- Resultado: reduce el error frente a una heurística rolling simple.

Este paper valida de forma independiente el diseño de `crosswalk.py` que ya
existe en el repo y da una plantilla metodológica concreta para "agregación
consciente de la alineación" en vez de inventar una desde cero.

---

## 2. El riesgo legal no es hipotético — hay que decidirlo ahora

No es solo un hallazgo de búsqueda: lo confirmé técnicamente. SofaScore prohíbe
**expresamente** en su ToS "robots, scripts, scraping, crawling, simulación o
navegación automatizada". Y tanto SofaScore como FBref, dentro de `soccerdata`,
exigen `seleniumbase` (control de navegador real) — indicio consistente de que
esos sitios están endurecidos contra el scraping simple, no una casualidad de
implementación.

A diferencia de BetPlay o Football-Data.co.uk, aquí `robots.txt` y el *rate
limiting* del protocolo existente **no bastan**: el problema no es solo de
cortesía técnica, es que el propio contrato de uso lo prohíbe por escrito.

**Recomendación:** no usar SofaScore ni WhoScored como fuente de alineaciones en
vivo. En su lugar:
- StatsBomb open-data para entrenamiento retrospectivo (licencia abierta).
- FBref vía `soccerdata` **solo** para features rolling históricas de partidos
  ya jugados (no como disparador — ya vimos que no puede serlo).
- Dejar como decisión explícita del usuario (§8) si quiere asumir el riesgo de
  una fuente en vivo tipo SofaScore, entendiendo que viola su ToS.

---

## 3. Fuga de datos: el riesgo metodológico central

Confirmado por la literatura encontrada (Wheatcroft, 2021; el paper de LaLiga
de §1.4): *"si las estadísticas del partido estuvieran disponibles de antemano,
se podrían hacer pronósticos muy informativos — pero nunca lo están antes del
partido"*. Dos formas concretas en que esto se colaría sin avisar:

1. **Usar una fuente cuyo dato "de alineación" es en realidad post-partido**
   (exactamente el caso de FBref `read_lineup()` que acabamos de descubrir). Si
   se usara ingenuamente como "la alineación del partido X" en vez de como
   archivo histórico de partidos ya jugados, contaminaría cualquier modelo.
2. **Tomar el valor "actual" de una métrica cuando en realidad hace falta el
   valor a la fecha del partido.** ClubElo lo permite bien —
   `read_by_date(date)` fija la fecha exacta — pero hay que usarlo así
   explícitamente y no con la fecha de hoy.

**Regla operativa (tomada del paper y de la práctica estándar), dos regímenes:**

- **Régimen histórico** (todo partido, se conozca o no la alineación):
  Elo a fecha, forma rolling de equipo a N partidos previos, fuerza de la
  plantilla completa.
- **Régimen operativo** (se activa en T-1h, cuando se confirma la alineación):
  agregación de rating/forma de los 11 titulares específicos y ausencias por
  lesión/sanción de quienes no juegan.

---

## 4. Catálogo de métricas candidatas

| Familia | Métrica | Fuente | Riesgo de fuga |
|---|---|---|---|
| Fuerza de equipo | Elo a fecha del partido | ClubElo | Bajo (fijando fecha) |
| Forma reciente | Goles/xG/tiros/córners rolling 5 partidos | FBref, Understat, o la propia Vía A ya cargada | Bajo si son estrictamente anteriores |
| Valor de plantilla | Valor de mercado total y medio | withqwerty/availability-data (Transfermarkt) | Bajo |
| Disponibilidad | `injury_burden`, jugadores sancionados/lesionados | withqwerty/availability-data | Bajo |
| Fuerza de la alineación anunciada | Suma/media de rating de los 11 titulares | SoFIFA/FIFA — **fuente pre-partido legal pendiente de resolver (§2)** | Alto si se usa mal |
| Contexto | Descanso entre partidos, local/visitante, congestión de calendario | Derivable del propio calendario ya cargado | Bajo |
| Árbitro | Tendencia histórica de tarjetas del colegiado | FBref/archivo | Bajo si es histórico |
| Precio de mercado (ya en el repo) | `p_devig`, EV, CLV | `oddsmath.py` | Se usa como referencia a batir, no como insumo del modelo propio |

---

## 5. Arquitectura de integración

1. `src/veee/lineup_model.py` (nombre provisional): estimador `p_lineup`,
   análogo a `p_shrunk`/`p_count` en `model.py`.
2. Generalizar `model.p_blend` de 2 a N fuentes (suma ponderada de logits — el
   pool logarítmico ya lo permite de forma natural).
3. Un evaluador nuevo en `capture.run_capture_cycle` que dispare específicamente
   en la ventana T-1h de `VENTANAS_H`, que ya existe.
4. **Ganancia barata e inmediata**: como ClubElo es histórico y consultable por
   fecha exacta, se puede enriquecer **retroactivamente** la matriz de ~26.600
   partidos que ya construye la Vía A, sin recolectar nada nuevo en vivo.

---

## 6. Fases

| Fase | Contenido | Depende de |
|---|---|---|
| **0** | Verificar desde una máquina con red: alcance real de ClubElo, cobertura de StatsBomb open-data en LaLiga actual, acceso al CSV de withqwerty/availability-data. Decidir la pregunta legal de §2 | Nada — barata y en paralelo |
| **1** | Enriquecer retroactivamente la Vía A con Elo a fecha de cada partido | Fase 0 |
| **2** | Features rolling desde FBref/Understat (requiere resolver la dependencia Selenium) | Fase 0 |
| **3** | Disponibilidad/lesiones desde withqwerty/availability-data | Fase 0 |
| **4** | Modelo de fuerza de alineación anunciada — el más incierto, depende de §2 | Fase 0 + decisión legal |
| **5** | Integración en `model.py` (N fuentes) y captura en vivo anclada a T-1h | Fases 1-4 |

---

## 7. Riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Ninguna fuente legal cubre alineaciones EN VIVO con antelación suficiente | Media | Alto | Fase 0 lo decide temprano; el régimen histórico (§3) sigue siendo útil sin él |
| FBref/Selenium no despliega en el entorno de captura (dependencias pesadas, ya confirmado que `soccerdata` falla al instalar sin ajustes) | Media | Medio | Priorizar ClubElo (sin navegador) primero; FBref es opcional |
| SoFIFA/FIFA ratings desincronizados de la plantilla real de la jornada | Media | Medio | Declarar la cadencia de actualización; no usar como única fuente de fuerza de jugador |
| Repetir el error de fuga ya detectado en FBref con otra fuente no auditada | Media | Crítico | Todo dato nuevo pasa por la pregunta explícita "¿esto existía ANTES del pitido inicial?" antes de usarse |

---

## 8. Decisiones para el usuario

1. **¿Asumir el riesgo de ToS de SofaScore/WhoScored para alineaciones en
   vivo, o renunciar a esa fuente y quedarse con el régimen histórico?**
2. **¿Empezar por la Fase 1** (Elo retroactivo sobre la Vía A — barato, usa
   datos que ya existen) **o por la captura en vivo** (más cara, con el riesgo
   legal de por medio)? Recomiendo Fase 1 primero.
3. **¿Instalar Selenium/Chromium** en el entorno de recolección para FBref, o
   limitarse a ClubElo y Understat (sin navegador) en una primera vuelta?
