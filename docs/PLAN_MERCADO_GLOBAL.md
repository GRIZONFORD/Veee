# Plan: Asimetría de Precios entre Mercados de Predicción y Casas de Apuestas

> Conversión del diseño propuesto (Polymarket + The Odds API + API-Football +
> SportsDataIO) en un plan ejecutable, con puertas de decisión, presupuesto de
> recursos y registro de riesgos.

---

## 0. Reencuadre: son tres preguntas, no una

El diseño propuesto mezcla tres preguntas de investigación que exigen métodos,
datos y estándares de prueba distintos. Separarlas es la primera decisión, porque
condiciona todo lo demás:

| | Pregunta | Método | Requiere ejecutabilidad | Riesgo |
|---|---|---|---|---|
| **Q1** | ¿Quién descubre el precio antes, Polymarket o las casas? | Hasbrouck / Gonzalo-Granger sobre series sincronizadas | No | Bajo |
| **Q2** | ¿Qué sede predice mejor? | Brier / log-loss / calibración | No | Medio (sesgo de selección) |
| **Q3** | ¿La divergencia ΔP genera alfa? | Paper trading + CLV + logit | Sí | Alto |

**Recomendación: Q1 es la columna vertebral de la tesis.** Es la pregunta con
literatura consolidada, con un método econométrico maduro, y —decisivo— **no
depende de que la divergencia sea explotable**. Q3 es un capítulo, no la tesis:
si no hay alfa, Q1 sigue en pie y el trabajo se sostiene.

El error estratégico sería construir toda la tesis sobre Q3 y descubrir en marzo
que no hay alfa (el resultado más probable) ni mercados solapados suficientes.

---

## 1. Hallazgo que condiciona el diseño

Calculado con el motor ya existente del repositorio, **antes** de recolectar nada:

```
mercado               vig     prop     shin    power   RANGO(pp)
1X2 equilibrado     0.048   0.4543   0.4587   0.4602       0.58
1X2 con favorito    0.052   0.6557   0.6681   0.6740       1.83
1X2 muy desigual    0.043   0.7991   0.8139   0.8220       2.30
O/U corners         0.053   0.4947   0.4945   0.4943       0.04
```

**La elección del método de de-vig mueve la probabilidad estimada una mediana de
1,21 pp (hasta 2,30 pp en libros desequilibrados).** Y el ΔP necesario para que
una apuesta cubra el margen es:

```
cuota 1.50 → ΔP > 3.17 pp     cuota 3.00 → ΔP > 1.59 pp
cuota 2.00 → ΔP > 2.38 pp     cuota 5.00 → ΔP > 0.95 pp
```

> **El error de medición inducido por una decisión metodológica es del mismo
> orden de magnitud que la señal que se busca.** Un ΔP de 2 pp en un partido con
> favorito claro puede ser íntegramente un artefacto de haber elegido el método
> proporcional en vez de Shin.

Esto no invalida el proyecto: lo disciplina. Se resuelve con dos decisiones que
deben tomarse **ex ante**:

### 1.1 Definición robusta de ΔP (contribución metodológica)

En lugar de `ΔP = p_poly − p_devig`, se define:

$$\Delta P^{\text{robusto}} = \begin{cases}
\operatorname{sgn}(\delta)\cdot\min_{m \in M} |p_{\text{poly}} - p^{(m)}_{\text{casa}}| & \text{si } \operatorname{sgn} \text{ coincide } \forall m \in M \\
0 & \text{en otro caso}
\end{cases}$$

con $M=\{$proporcional, aditivo, potencia, Shin$\}$. Es decir: **la señal solo
existe si los cuatro métodos coinciden en el signo**, y su magnitud es la más
conservadora del conjunto. Cualquier ΔP que dependa del método elegido se anula
por construcción. Cuesta ~30 líneas sobre `src/veee/oddsmath.py`, que ya
implementa los cuatro estimadores.

### 1.2 Estratificación por equilibrio del libro

Reportar todos los resultados separando libros equilibrados (favorito < 0,65) de
desequilibrados, donde la incertidumbre del de-vig se dispara. Es una variable de
control obligatoria en la regresión, no una nota al pie.

### 1.3 Coste de capital en Polymarket

Un precio $p$ en Polymarket **no es una probabilidad**: es el valor presente de
un pago de $1 condicionado. Con USDC inmovilizado hasta la resolución:

$$q = p \cdot (1+r)^{T/365}$$

```
  3 días → +0.04%      180 días → +2.44%
 30 días → +0.40%      365 días → +5.00%
```

Despreciable para mercados por partido. **Material —y probablemente mayor que el
alfa— para mercados de temporada** (ganador de liga, máximo goleador). Y aquí
está la trampa: es precisamente en esos mercados de larga duración donde
Polymarket concentra su liquidez en fútbol. Sin esta corrección, un estudio sobre
outrights confundiría el coste de capital con ineficiencia.

---

## 2. Fase 0 — Viabilidad (2 semanas) · **PUERTA GO/NO-GO**

**Esta fase decide si el proyecto existe.** Todo el diseño cruzado depende de que
haya mercados solapados suficientes, y eso **no está verificado**. No pude
comprobarlo desde el entorno de desarrollo (la política de red bloquea las cinco
APIs: `clob.polymarket.com`, `gamma-api.polymarket.com`, `api.the-odds-api.com`,
`v3.football.api-sports.io`, `api.sportsdata.io` — todas devuelven sin conexión).

### Riesgo principal, declarado sin adornos

Polymarket es un mercado de predicción generalista. Su liquidez en deportes se
concentra en eventos de alto perfil (Mundial, finales de Champions, ganador de
liga). **Es dudoso que exista un mercado 1X2 líquido para un Getafe–Alavés de
jornada 14, y prácticamente seguro que no existen mercados de córneres o
tarjetas.** Si el solapamiento con LaLiga por partido es escaso, el estudio
cruzado por partido no es viable tal como está planteado.

### Entregable de la Fase 0

```
scripts/feasibility.py --sport soccer --season 2025-26
```

Un censo, no un análisis. Debe responder con números:

1. **Cobertura.** ¿Cuántos eventos de LaLiga tienen mercado en Polymarket? ¿Qué
   tipos (1X2, outright, props)? ¿Con qué antelación se abren?
2. **Liquidez.** Para cada mercado solapado: profundidad del libro, *spread*
   bid-ask, volumen diario, número de operaciones. **Un mercado con spread de
   4 pp no puede usarse para medir un ΔP de 2 pp.**
3. **Solapamiento temporal.** ¿Hay cotización simultánea en ambas sedes durante
   una ventana suficiente para estimar lead-lag? Un mercado que solo se activa
   2 horas antes no sirve para price discovery.
4. **Resolución.** ¿Cómo y cuándo resuelve Polymarket? ¿Qué oráculo (UMA)?
   ¿Cuánto tarda? Afecta al coste de capital y a la fecha de liquidación.
5. **Coste de cuota API.** Consumo real por barrido (ver §7).

### Criterios de decisión

| Resultado del censo | Decisión |
|---|---|
| ≥ 100 partidos solapados con spread < 2 pp y ventana > 24 h | **GO** — diseño por partido, LaLiga |
| Solapamiento escaso en LaLiga pero suficiente en ligas top-5 / Champions | **GO con pivote** — ampliar a "mercado global", que es lo que el título ya sugiere |
| Solo mercados outright de temporada | **GO con rediseño** — estudio de outrights, con corrección de coste de capital como eje central |
| < 30 mercados solapados con liquidez utilizable | **NO-GO** — replegar al diseño de una sola sede (multi-casa vía The Odds API), que sigue siendo una tesis válida |

**El plan de repliegue no es un fracaso.** The Odds API sola ya permite un
estudio sólido de eficiencia entre casas (Pinnacle como referencia *sharp* frente
a casas *soft*), y resuelve el problema de identificación del estudio anterior.

---

## 3. Fase 1 — Arquitectura e ingesta (3 semanas)

### 3.1 Sobre la POO propuesta: de acuerdo en parte

**Adoptar** la jerarquía `BaseDataConnector(abc.ABC)`. Con cuatro fuentes
heterogéneas el polimorfismo se gana su sitio: cada conector normaliza a un
contrato común y el orquestador queda agnóstico. Es la decisión correcta.

**No adoptar** `EconomicEngine` como clase. Tal como está propuesta, contiene
solo `@staticmethod`: no tiene estado, no tiene identidad, no se instancia. Eso
es un módulo con sintaxis de clase. En Python, un módulo de funciones puras ya
proporciona el espacio de nombres, y las funciones puras son más fáciles de
probar, componer y paralelizar. `src/veee/oddsmath.py` ya es exactamente eso, con
83 pruebas detrás. **Envolverlo en una clase sería refactorizar hacia atrás.**

La regla que sí conviene aplicar: POO donde hay **estado y polimorfismo**
(conectores, gestor de BD, presupuesto de cuotas); funciones puras donde hay
**matemática sin estado** (de-vig, ΔP, Brier, contrastes).

### 3.2 Defectos a corregir del código propuesto

| Defecto | Consecuencia | Corrección |
|---|---|---|
| `datetime.utcnow()` | Devuelve *naive*; comparar sedes en husos distintos da desalineación silenciosa. Obsoleto desde 3.12 | `datetime.now(timezone.utc)` |
| Precio de Polymarket tratado como probabilidad | Ignora spread y coste de capital | Registrar `best_bid`, `best_ask`, `mid`, `last`; aplicar §1.3 |
| Sin control de cuota API | The Odds API son 500 créditos/mes; un barrido ingenuo los agota en días | Clase `QuotaBudget` con contabilidad persistente |
| Sin límite de tasa ni reintentos | Bloqueo por parte del proveedor | Reutilizar `PoliteSession` del repo |
| Conexión SQLite por operación, *commit* por fila | Inviable a nivel de libro de órdenes | WAL + inserción por lotes |
| Sin clave de idempotencia | Reejecutar duplica filas | IDs deterministas (ya resuelto en `database.py`) |
| Conectores devuelven datos simulados | Aparenta funcionar sin datos reales | *Preflight* que verifique contrato contra la API real |

### 3.3 Módulos nuevos

```
src/veee/connectors/
  base.py          BaseDataConnector(ABC) · contrato común · QuotaBudget
  polymarket.py    Gamma (descubrimiento) + CLOB (libro, mid, spread, trades)
  odds_api.py      Multi-casa; Pinnacle marcado como referencia sharp
  api_football.py  Fixtures, estadísticas, xG (verificar cobertura real)
  sportsdataio.py  Validación cruzada de líneas de apertura
src/veee/crosswalk.py   Emparejamiento de eventos entre sedes
src/veee/discovery.py   Sincronización y métricas de price discovery
```

### 3.4 El emparejamiento de eventos es el punto crítico

`crosswalk.py` decide si el estudio tiene datos. Cada sede nombra los equipos de
forma distinta ("Atlético de Madrid" / "Atletico Madrid" / "Atl. Madrid"), y
Polymarket titula en lenguaje natural ("Will Barcelona beat Girona?").

Estrategia en cascada, con verificación humana obligatoria:
1. Emparejamiento exacto por (fecha, equipos canonizados) — reutiliza
   `ALIAS_EQUIPOS` del repo.
2. Difuso (`rapidfuzz`) con umbral alto sobre los residuales.
3. **Revisión manual de todo emparejamiento difuso**, registrada en una tabla de
   correspondencias versionada.

Un emparejamiento erróneo no produce un error: produce un ΔP espurio enorme.
Es el fallo silencioso más peligroso del diseño. El *preflight* del repo ya
implementa esta comprobación de cruce para dos fuentes; se extiende a cuatro.

---

## 4. Fase 2 — Motor econométrico (2 semanas)

Extensión de `oddsmath.py` y `econometrics.py`, ya existentes y probados:

- `delta_p_robusto()` — §1.1
- `ajuste_coste_capital()` — §1.3
- `prob_polymarket()` — mid vs last vs micro-precio ponderado por profundidad;
  el micro-precio $\frac{p_{ask}\cdot v_{bid} + p_{bid}\cdot v_{ask}}{v_{bid}+v_{ask}}$
  es preferible al punto medio cuando el libro está desequilibrado
- `brier_descompuesto()` — descomposición de Murphy en **fiabilidad,
  resolución e incertidumbre**. Un Brier agregado no distingue "mal calibrado" de
  "poco informativo", y esa distinción es justamente la pregunta Q2

---

## 5. Fase 3 — Price discovery (Q1, 4 semanas) · **núcleo de la tesis**

### 5.1 El método propuesto necesita corrección

La causalidad de Granger sobre precios en niveles es **espuria**: los precios son
casi martingalas, con raíz unitaria. Aplicar Granger sin tratar la no
estacionariedad produce rechazos sistemáticos sin contenido.

Diseño correcto, que es además el estándar de la literatura de microestructura:

1. **Sincronización.** Rejilla temporal común (1 o 5 min) con última observación
   arrastrada. Documentar el sesgo por asincronía: las casas actualizan de forma
   irregular y eso atenúa las estimaciones de lead-lag.
2. **Raíces unitarias.** ADF y KPSS sobre cada serie (en logit, no en niveles:
   el precio está acotado en [0,1] y el logit lo lleva a la recta real).
3. **Cointegración.** Dos precios del *mismo* evento deben estar cointegrados con
   vector $(1,-1)$. Es una **restricción contrastable**, no un supuesto: si se
   rechaza, las sedes están valorando cosas distintas y hay que revisar el
   emparejamiento antes que la economía.
4. **VECM** y, sobre él:
   - **Information Share de Hasbrouck** (cotas superior e inferior; reportar
     ambas, no solo la media)
   - **Component Share de Gonzalo-Granger**
   - Granger sobre los residuos del VECM, ya estacionarios — aquí sí es válido
5. **Robustez.** Submuestras por liquidez, por antelación al evento y por tipo de
   mercado.

### 5.2 Hipótesis y su lectura económica

| | Hipótesis | Interpretación si se confirma |
|---|---|---|
| H1 | IS(Polymarket) > 0,5 | El mercado descentralizado lidera: el capital especulativo agrega información antes que el creador de mercado |
| H2 | IS(Pinnacle) > IS(casas *soft*) | Jerarquía informativa dentro de TradFi, como predice la literatura |
| H3 | IS(Polymarket) crece con la liquidez | El liderazgo es función de la profundidad, no de la tecnología |

**H3 es la más interesante y la menos obvia.** Si el liderazgo de Polymarket
depende de su liquidez, entonces la ventaja no es "blockchain vs TradFi" sino
simplemente profundidad de mercado — y eso desinfla buena parte de la narrativa
DeFi con evidencia propia. Es el tipo de resultado que distingue una tesis.

---

## 6. Fase 4 — Alfa (Q3, 4 semanas)

Reutiliza **íntegro** el aparato ya construido y probado: registro ex ante
inmutable, escalera de captura, línea de cierre, CLV, contrastes robustos.

### 6.1 Lo que este diseño resuelve del estudio anterior

El problema de identificación del estudio BetPlay–Linemate era que $\hat p$ salía
de una señal débil. Aquí **el precio de Polymarket es un conjunto de información
genuinamente independiente del precio de la casa**. Ese es el argumento más
fuerte a favor de este pivote y debe declararse así en la tesis.

### 6.2 Simetría obligatoria

Si Polymarket resulta *menos* eficiente que Pinnacle (plausible con liquidez
fina), entonces ΔP mide sobre todo ruido de Polymarket, y el alfa está en apostar
**contra** Polymarket — operación que sí es ejecutable, porque en Polymarket se
puede tomar el lado contrario. **El diseño debe contrastar ambas direcciones**;
asumir a priori que la ineficiencia está en la casa tradicional sería petición de
principio.

### 6.3 Contraste principal: CLV, no ROI

Por potencia estadística: detectar un ROI del 3 % exige ~6.870 apuestas
(calculado en el repo). El CLV se mide sin el ruido del resultado. Se mide contra
la línea de cierre de **Pinnacle**, referencia *sharp* estándar de la literatura.

### 6.4 Límites al arbitraje — sección obligatoria

Una divergencia estadística no es alfa ejecutable:

- No se puede vender en corto una apuesta en una casa.
- Polymarket aplica restricciones geográficas; verificar el acceso legal desde
  Colombia **antes** de plantear ejecutabilidad.
- Coste de gas, *spread*, y capital inmovilizado hasta la resolución.
- *Limiting*: la casa recorta el stake del apostador rentable.

La tesis debe distinguir **ineficiencia estadística** de **explotabilidad
económica**. Es la distinción que da madurez al trabajo.

---

## 7. Presupuesto de cuota API — restricción vinculante

**The Odds API: 500 créditos/mes en el plan gratuito**, y el coste es
`regiones × mercados` por petición. Un barrido ingenuo (10 casas × 380 partidos ×
escalera de 10 ventanas) agota la cuota en días. Consecuencias de diseño:

- Clase `QuotaBudget` con contabilidad persistente y **freno duro** antes de
  agotar la cuota; sin ella, la recolección muere a mitad de temporada y deja una
  muestra truncada de forma no aleatoria.
- Escalera de ventanas **adaptativa**: densa solo cerca del cierre (donde está el
  valor informativo) y para los partidos efectivamente solapados con Polymarket.
- Polymarket y API-Football tienen sus propios límites; verificarlos en Fase 0.
- Presupuestar el plan de pago desde el principio si la Fase 0 dice GO: el coste
  de la cuota es menor que el de una temporada perdida.

---

## 8. Registro de riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Polymarket no cubre LaLiga por partido | **Alta** | **Crítico** | Fase 0 con puerta GO/NO-GO; plan de repliegue multi-casa |
| ΔP dominado por el método de de-vig | **Alta** | Alto | ΔP robusto (§1.1) + estratificación |
| Spread de Polymarket > señal | Media | Alto | Filtro de liquidez mínima en Fase 0 |
| Emparejamiento erróneo de eventos | Media | **Crítico** | Cascada + revisión manual + preflight |
| Cuota API agotada a mitad de temporada | Media | Alto | `QuotaBudget` con freno duro |
| Muestra insuficiente para Q3 | **Alta** | Medio | Q1 como núcleo; CLV como contraste principal |
| Acceso a Polymarket restringido geográficamente | Media | Medio | Verificar en Fase 0; solo afecta a Q3 |
| Cambio de API sin aviso | Media | Medio | `health_check` ya implementado |

---

## 9. Qué se reutiliza (≈60 % ya construido y probado)

| Ya existe | Se reutiliza para |
|---|---|
| `oddsmath.py` (4 métodos de de-vig, EV, CLV, Kelly) | Base de ΔP robusto |
| `econometrics.py` (t, HAC, cluster, bootstrap, logit, FDR, potencia) | Q3 sin cambios |
| `capture.py` (escalera, cierre, salud) | Ingesta multi-sede |
| `database.py` (inmutabilidad ex ante, idempotencia) | Persistencia |
| `preflight.py` (validación de cruce) | Extender a 4 fuentes |
| `simulate.py` (DGP, potencia) | Añadir DGP de dos sedes con lead-lag conocido |
| 83 pruebas | Red de seguridad del refactor |

**Nuevo:** conectores, crosswalk, price discovery (VECM/Hasbrouck), presupuesto
de cuota.

---

## 10. Cronograma y puertas

```
Sem  1-2   FASE 0  Viabilidad ─────────────► PUERTA GO/NO-GO
Sem  3-5   FASE 1  Conectores + crosswalk ─► PUERTA: cruce verificado
Sem  6-7   FASE 2  Motor econométrico
Sem  8-11  FASE 3  Price discovery (Q1)  ◄── núcleo de la tesis
Sem 12-15  FASE 4  Alfa (Q3) + recolección en paralelo desde sem. 6
Sem 16-18  FASE 5  Redacción
```

**La recolección debe empezar en la semana 6**, en cuanto los conectores pasen el
preflight. Los precios históricos no se pueden reconstruir a posteriori: cada
semana sin capturar es muestra perdida para siempre.

### Pre-registro

Antes del primer dato de la Fase 4, congelar en `docs/PREREGISTRO.md`: umbral de
ΔP, ventana de colocación, definición de la sede de referencia, criterio de
liquidez mínima y regla de parada por calendario. Los parámetros no se tocan
después de ver resultados.

---

## 11. Decisiones que requieren su criterio

1. **Alcance geográfico.** "Mercado global" sugiere ampliar más allá de LaLiga.
   Ampliar mejora el solapamiento con Polymarket (más eventos de alto perfil)
   pero introduce heterogeneidad entre ligas. Recomiendo: LaLiga como núcleo,
   ligas top-5 + Champions como extensión si la Fase 0 lo exige.
2. **Peso relativo Q1 vs Q3.** Recomiendo Q1 como columna vertebral, por robustez
   frente al resultado. Si la tesis debe versar sobre alfa, hay que asumir el
   riesgo de un resultado nulo y declararlo en el pre-registro.
3. **Presupuesto.** ¿Hay margen para planes de pago de API? Condiciona
   directamente la densidad de la escalera de captura y, con ella, la calidad de
   las estimaciones de lead-lag.
