# Plan de Acción — Asimetría de Precios y Búsqueda de Alfa · **Coste cero**

> Versión 2. Rediseñado bajo la restricción de coste cero, que resulta no ser una
> limitación sino una mejora: obliga a abandonar la recolección prospectiva como
> fuente principal y a apoyarse en el archivo histórico gratuito, que tiene entre
> **6 y 10 veces más potencia estadística**.

---

## 0. Lo que cambia al imponer coste cero

El plan anterior descansaba en recolectar una temporada de datos prospectivos.
Con precios gratuitos eso es inviable —y, sobre todo, **era una mala idea desde el
principio**:

```
escenario                                              n      MDE ROI
1 temporada LaLiga, 2 apuestas/partido               760        9.02%
Football-Data.co.uk: LaLiga, 10 temporadas         3,800        4.03%
  + 7 ligas europeas, 10 temporadas                26,600       1.52%
  x 3 selecciones (1X2) por partido                79,800       0.88%

Alfa documentado en la literatura: 1-3%
n necesario para detectar ROI del 2%:              15,456
```

**Una temporada prospectiva solo detecta un ROI del 9%.** Nadie sostiene que
exista alfa de esa magnitud en un mercado maduro: el diseño estaba condenado a un
resultado nulo no informativo. El archivo histórico gratuito baja el efecto
mínimo detectable a **0,88 %**, por debajo del alfa que la literatura documenta.

> La restricción presupuestaria corrigió un error de diseño. El cuello de botella
> nunca fue el dinero: era la **potencia estadística**.

Consecuencia estructural: **la columna vertebral pasa a ser el archivo histórico**
(disponible hoy, sin esperar), y la recolección prospectiva queda como la vía
novedosa, no como la fuente principal.

---

## 1. Inventario de fuentes gratuitas

### 1.1 Se adoptan

| Fuente | Coste | Aporta | Límite |
|---|---|---|---|
| **Football-Data.co.uk** | Gratis, sin clave | CSV por liga y temporada: cuotas 1X2, O/U 2.5 y hándicap asiático de varias casas (**incluida Pinnacle**), apertura y cierre, más córneres, tarjetas, tiros y faltas | Sin límite; descarga directa |
| **Polymarket** (Gamma + CLOB) | **Gratis, sin clave** | Libro de órdenes, mid, spread, histórico de precios | Uso razonable |
| **The Odds API** | 500 créditos/mes | Cuotas en vivo multi-casa, incluida Pinnacle | Cuota estricta; **sin histórico en el plan gratuito** |
| **football-data.org** | Gratis (10 pet./min) | Calendario y resultados para liquidar | Competiciones limitadas |
| **API-Football** | 100 pet./día | Estadísticas de partido | xG probablemente de pago: **verificar** |

### 1.2 Se descarta

**SportsDataIO** — solo prueba gratuita, después de pago. No aporta nada que las
anteriores no cubran. Fuera del plan.

### 1.3 Football-Data.co.uk es el hallazgo que reordena el proyecto

Es el conjunto de datos estándar de la literatura académica sobre eficiencia en
apuestas. Gratuito, sin registro, y contiene justo lo que el estudio necesitaba:

- Cuotas de **múltiples casas** por partido → la "asimetría de cuotas" del título
  es medible **directamente y con N enorme**, sin esperar una temporada.
- **Pinnacle** incluida → referencia *sharp* de la literatura.
- **Cuotas de cierre** → el CLV, contraste de mayor potencia, se calcula sobre
  decenas de miles de partidos en vez de sobre cientos.
- **Córneres y tarjetas observados** (`HC`, `AC`, `HY`, `AY`) → los micro-mercados
  del estudio original, con resultados reales aunque sin sus cuotas.

### 1.4 Lo que hay que verificar en la Fase 0 (no pude comprobarlo)

La política de red del entorno bloquea las cinco APIs (`000` en todas). Estos
puntos vienen de conocimiento previo y **deben confirmarse antes de construir**:

1. **Nombres de columna y su disponibilidad por temporada.** Las columnas de
   cierre (`PSCH`, `B365CH`…) probablemente **no existen en temporadas
   antiguas**, y las columnas agregadas cambiaron de esquema (`BbMxH`/`BbAvH` →
   `MaxH`/`AvgH`) hacia 2019. **El esquema del CSV no es estable entre
   temporadas**: el cargador necesita un mapa de columnas por temporada, no un
   `read_csv` único. Es la principal tarea de ingeniería de la Vía A.
2. Si el CLV histórico solo existe desde ~2019, la muestra para CLV es menor que
   la muestra total. Cuantificarlo determina el alcance real.
3. Cobertura de fútbol en Polymarket (§4.1).
4. Coste real en créditos por petición en The Odds API.
5. Si API-Football sirve xG en el plan gratuito.

---

## 2. Arquitectura de dos vías

```
        VÍA A — ARCHIVO (columna vertebral)          VÍA B — PROSPECTIVA (novedad)
   ┌────────────────────────────────────┐      ┌────────────────────────────────────┐
   │ Football-Data.co.uk                │      │ Polymarket CLOB  ×  The Odds API   │
   │ 7 ligas × 10 temporadas            │      │ Descubrimiento de precios          │
   │ n ≈ 26.600 partidos                │      │ n pequeño, alto valor novedoso     │
   │ Disponible HOY                     │      │ Requiere temporada en curso        │
   │ Riesgo: BAJO                       │      │ Riesgo: ALTO (cobertura)           │
   └────────────────────────────────────┘      └────────────────────────────────────┘
                    │                                          │
                    └──────────────┬───────────────────────────┘
                          Mismo motor econométrico
```

**La Vía A garantiza que hay tesis.** Se puede completar entera en 6 semanas con
datos que ya existen. La Vía B aporta la contribución novedosa, y si su Fase 0
sale NO-GO, la tesis sigue en pie.

Este desacoplamiento es la mejora principal frente al plan anterior, que era todo
o nada sobre una puerta GO/NO-GO.

---

## 3. Vía A — Archivo histórico (semanas 1–6)

### 3.1 Preguntas, todas de gran N y coste cero

| | Pregunta | Contraste |
|---|---|---|
| A1 | ¿Persiste el sesgo favorito-*longshot* por tramo de cuota? | ROI por decil de probabilidad implícita |
| A2 | **¿La dispersión entre casas predice el resultado?** | Logit de acierto sobre la desviación típica de las cuotas entre casas |
| A3 | ¿Bate alguien la línea de cierre de Pinnacle? | CLV medio por casa, con EE agrupados |
| A4 | ¿El desplazamiento apertura→cierre predice? | Regresión de eficiencia con el desplazamiento como regresor |
| A5 | ¿La eficiencia mejora con los años? | Interacción con la temporada |

**A2 es la "asimetría de cuotas" del enunciado**, medida directamente: cuando las
casas discrepan entre sí sobre el mismo partido, ¿está la discrepancia informada
o es ruido? Es una pregunta nítida, con N de decenas de miles y respuesta
garantizada en cualquier dirección.

### 3.2 Ventaja metodológica decisiva

El estudio original tenía un problema de identificación: $\hat p$ salía del mismo
libro sobre el que se apostaba, y el EV era idénticamente $-m/(1+m)$. Aquí
**la cuota de una casa distinta es un conjunto de información genuinamente
independiente**. El problema desaparece por diseño.

### 3.3 Trampas del histórico que hay que declarar

- **Sesgo de supervivencia**: solo aparecen las casas que seguían operando.
- **Sin ejecutabilidad**: la cuota del CSV pudo no estar disponible al público en
  ese momento, ni admitir stake relevante. El histórico mide **ineficiencia
  estadística**, nunca alfa ejecutable. Debe decirse explícitamente.
- ***Data snooping***: con 7 ligas × 10 temporadas × varias casas es trivial
  encontrar significación por azar. Corrección FDR obligatoria (ya implementada)
  y **partición temporal**: calibrar en las temporadas antiguas, validar en las
  recientes, sin volver atrás.

### 3.4 Ingeniería

```
scripts/fetch_archive.py     # descarga y cachea los CSV por liga y temporada
src/veee/archive.py          # mapa de columnas POR TEMPORADA + normalización
```

El mapa de columnas por temporada es el núcleo: sin él, un `read_csv` ingenuo
produce columnas ausentes silenciosas y una muestra sesgada por omisión. Debe ir
acompañado de un informe de cobertura por temporada y casa.

---

## 4. Vía B — Prospectiva: Polymarket × The Odds API (semanas 7–15)

### 4.1 Fase 0 de viabilidad — puerta GO/NO-GO (semana 7, 3 días)

Sigue sin verificarse que Polymarket cubra LaLiga por partido. Es dudoso que haya
mercado líquido para un Getafe–Alavés y casi seguro que no hay córneres ni
tarjetas. El censo (`scripts/feasibility.py`) debe medir cobertura, **spread**,
profundidad, ventana de cotización simultánea y mecanismo de resolución.

**Un mercado con spread de 4 pp no sirve para medir un ΔP de 2 pp.** El filtro de
liquidez es un criterio de inclusión, no un detalle.

| Resultado | Decisión |
|---|---|
| ≥100 partidos con spread < 2 pp y ventana > 24 h | GO — diseño por partido |
| Solapamiento solo en ligas top / Champions | GO con pivote a "mercado global" |
| Solo *outrights* de temporada | GO con rediseño — corrección de coste de capital como eje |
| < 30 mercados utilizables | **NO-GO** — la Vía A ya sostiene la tesis; Vía B pasa a "líneas futuras" |

Polymarket es gratis y sin clave, de modo que **el censo cuesta cero**: no hay
razón para no ejecutarlo en la semana 1 en paralelo con la Vía A.

### 4.2 Descubrimiento de precios (el contenido de vanguardia)

Cadena metodológica, corrigiendo el diseño propuesto:

1. Sincronización en rejilla de 5 min con última observación arrastrada.
2. Raíz unitaria (ADF + KPSS) **en escala logit**, no en niveles: el precio está
   acotado en [0,1].
3. Cointegración con vector $(1,-1)$ **como restricción contrastable**: dos
   precios del mismo evento deben cointegrar. Si se rechaza, revisar primero el
   emparejamiento de eventos, no la economía.
4. VECM y sobre él **Information Share de Hasbrouck** (reportar cotas superior e
   inferior) y **Component Share de Gonzalo-Granger**.
5. Granger **solo sobre los residuos del VECM**, ya estacionarios.

> La causalidad de Granger sobre precios en niveles es espuria: los precios son
> casi martingalas con raíz unitaria y el contraste rechaza sistemáticamente sin
> contenido informativo. Es la corrección técnica más importante al diseño
> original.

**Hipótesis más interesante (H3):** que el liderazgo de Polymarket dependa de su
liquidez y no de su tecnología. Si se confirma, la ventaja no es "DeFi frente a
TradFi" sino simple profundidad de mercado — y eso desinfla buena parte de la
narrativa descentralizada con evidencia propia.

### 4.3 El precio de Polymarket no es una probabilidad

- Registrar `best_bid`, `best_ask`, `mid` y **micro-precio** ponderado por
  profundidad: $\frac{p_{ask} v_{bid} + p_{bid} v_{ask}}{v_{bid}+v_{ask}}$,
  preferible al punto medio con libro desequilibrado.
- **Coste del capital inmovilizado**: $q = p\,(1+r)^{T/365}$ → +0,04 % a 3 días
  pero **+2,44 % a 180 días**, magnitud mayor que el alfa típico. Despreciable por
  partido; **decisivo en *outrights***, que es justamente donde Polymarket
  concentra liquidez en fútbol. Sin esta corrección se confundiría coste de
  capital con ineficiencia.

---

## 5. Presupuesto de cuota — The Odds API (500 créditos/mes)

El coste es `regiones × mercados` **por petición**, y una petición devuelve todos
los partidos del deporte: el coste **no escala con el número de partidos**, sino
con la frecuencia de sondeo. Esto lo cambia todo respecto al plan anterior.

```
Sondeo diario uniforme:
  solo h2h            16 llamadas/día  ->  480 cred/mes   OK
  h2h + totals         8 llamadas/día  ->  480 cred/mes   OK
  h2h+totals+spreads   8 llamadas/día  ->  720 cred/mes   EXCEDE

Concentrando SOLO en jornadas (8 jornadas/mes):
  solo h2h            14 llamadas/jornada ->  112 cred/mes  OK
  h2h+totals+spreads  14 llamadas/jornada ->  336 cred/mes  OK
```

**Decisión: escalera concentrada en jornadas, 3 mercados, ~14 llamadas por
jornada = 336 créditos/mes**, con 164 de reserva para reintentos y días
excepcionales. Cabe holgadamente en el plan gratuito.

Implementación obligatoria: clase `QuotaBudget` con contabilidad persistente y
**freno duro**. Sin ella la recolección muere a mitad de temporada y deja una
muestra truncada de forma no aleatoria — un sesgo peor que no tener los datos.
La API devuelve los créditos restantes en cabeceras: registrarlas en cada llamada
y reconciliar con el contador propio.

---

## 6. Infraestructura gratuita

- **GitHub Actions**: minutos ilimitados en repositorio público. A diferencia de
  BetPlay, ni Polymarket ni The Odds API tienen bloqueo geográfico, de modo que
  **el workflow ya existente sirve como host de captura sin coste** y sin
  depender de una máquina encendida. Es una mejora práctica real frente al plan
  anterior.
- Limitación asumida: el planificador de Actions se retrasa 5–15 min, lo que
  degrada la precisión de la línea de cierre prospectiva. **No importa**, porque
  el CLV preciso viene de la Vía A (cierres históricos reales). División del
  trabajo deliberada.
- Persistencia: SQLite versionado en el repositorio. Copia de seguridad gratuita
  e implícita en el historial de git.
- Cómputo econométrico: local. `statsmodels` cubre VECM, Johansen, ADF/KPSS y
  Granger sin coste.

---

## 7. Decisiones metodológicas que se mantienen

Vigentes de la versión 1, sin cambios:

- **ΔP robusto**: la señal solo existe si los cuatro métodos de de-vig coinciden
  en signo; magnitud, la más conservadora. Motivo: el método mueve la
  probabilidad una mediana de **1,21 pp** (hasta 2,30 pp en libros
  desequilibrados) mientras que el ΔP necesario para cubrir el margen va de 0,95
  a 3,17 pp. **El error metodológico es del orden de la señal.**
- **Estratificación por equilibrio del libro** (favorito < 0,65) como variable de
  control obligatoria.
- **POO donde hay estado y polimorfismo** (conectores, `QuotaBudget`, gestor de
  BD); **funciones puras donde hay matemática sin estado**. `EconomicEngine` con
  solo `@staticmethod` es un módulo con sintaxis de clase; `oddsmath.py` ya
  cumple esa función con 83 pruebas detrás.
- Correcciones de código: `datetime.now(timezone.utc)` en vez de `utcnow()`
  (obsoleto y *naive*), límite de tasa, WAL e inserción por lotes, IDs
  deterministas.

---

## 8. Cronograma

```
Sem 1     VÍA A  Cargador de archivo + mapa de columnas por temporada
          VÍA B  Censo de viabilidad de Polymarket (gratis, en paralelo)  ──► PUERTA
Sem 2-3   VÍA A  Normalización, informe de cobertura, control de calidad
Sem 4-6   VÍA A  Econometría A1-A5  ◄── AQUÍ YA HAY TESIS DEFENDIBLE
Sem 7     VÍA B  Conectores + crosswalk + QuotaBudget
Sem 8     VÍA B  Preflight y arranque de captura prospectiva
Sem 9-14  VÍA B  Captura continua (en segundo plano) + price discovery
Sem 15-18 Redacción
```

**Punto crítico: en la semana 6 ya existe una tesis completa y defendible**, con
26.600 partidos y potencia adecuada. Todo lo posterior es contribución adicional,
no rescate. Ésa es la diferencia con el plan anterior, donde un NO-GO en la
semana 2 dejaba el proyecto sin nada.

**La captura prospectiva debe arrancar en la semana 8**: los precios no se
reconstruyen a posteriori y cada semana sin capturar es muestra perdida.

### Pre-registro

Antes de la primera regresión de la Vía A, congelar en `docs/PREREGISTRO.md`: la
partición temporal calibración/validación, el umbral de ΔP, el criterio de
liquidez mínima y la regla de parada por calendario. Con 26.600 observaciones y
libertad de especificación, el *data snooping* es el riesgo dominante.

---

## 9. Riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Esquema del CSV inestable entre temporadas | **Alta** | Medio | Mapa de columnas por temporada + informe de cobertura |
| Cuotas de cierre ausentes en temporadas antiguas | **Alta** | Medio | Restringir el análisis CLV; declarar la submuestra |
| Polymarket no cubre LaLiga por partido | **Alta** | **Bajo** (antes: crítico) | Vía A independiente; Vía B pasa a líneas futuras |
| *Data snooping* con N grande | **Alta** | Alto | FDR + partición temporal + pre-registro |
| ΔP dominado por el método de de-vig | Alta | Alto | ΔP robusto + estratificación |
| Emparejamiento erróneo de eventos | Media | Crítico (Vía B) | Cascada + revisión manual + preflight |
| Cuota API agotada | Baja | Alto | `QuotaBudget` con freno duro (336 de 500) |
| Sesgo de supervivencia de casas | Media | Medio | Declarar; análisis por cohortes |

---

## 10. Lo que hace falta decidir

1. **Alcance de ligas en la Vía A.** Recomiendo las 7 principales europeas
   (España, Inglaterra ×2, Italia, Alemania, Francia, Portugal): maximiza N y
   permite contrastar si la eficiencia varía con la liquidez de la liga —
   pregunta interesante en sí misma. LaLiga sola basta si prefiere foco.
2. **Profundidad temporal.** 10 temporadas dan n≈26.600. Más años añaden N pero
   también heterogeneidad estructural (el mercado de 2010 no es el de 2025). Si
   la eficiencia mejora con el tiempo (A5), mezclar épocas confunde el promedio.
3. **Peso Vía A / Vía B.** Recomiendo 60/40 a favor de A: garantiza resultados y
   deja espacio para la contribución novedosa.

---

## 11. Qué reutiliza (≈70 % ya construido y probado)

| Existe | Uso |
|---|---|
| `oddsmath.py` — 4 de-vig, EV, CLV, Kelly | Base de ΔP robusto y de toda la Vía A |
| `econometrics.py` — t, HAC, cluster, bootstrap, logit, FDR, potencia | Vías A y B sin cambios |
| `database.py` — inmutabilidad ex ante, idempotencia | Persistencia |
| `capture.py` — escalera, cierre, salud | Vía B |
| `preflight.py` — validación de cruce | Extender a las fuentes nuevas |
| `.github/workflows/captura.yml` | Host gratuito de la Vía B |
| 83 pruebas | Red de seguridad |

**Nuevo:** `archive.py` (mapa de columnas por temporada), conectores Polymarket y
Odds API, `QuotaBudget`, `discovery.py` (VECM/Hasbrouck).
