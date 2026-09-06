# Esquema del Paper Académico

**Título propuesto:** *¿Contienen las tendencias estadísticas públicas información
no incorporada a los precios? Una prueba de eficiencia semifuerte en el mercado
de apuestas de LaLiga*

**Extensión orientativa:** 8.000–10.000 palabras · **Clasificación JEL:** G14
(Eficiencia de mercado), D82 (Asimetría de información), L83 (Deportes), C53
(Predicción)

---

## 1. Introducción (~1.200 palabras)

**Gancho.** Los mercados de apuestas deportivas son el laboratorio natural de la
Hipótesis de Mercados Eficientes: a diferencia de una acción, **cada activo tiene
un valor terminal observable y una fecha de vencimiento conocida**. No hay valor
residual, ni horizonte infinito, ni problema de la hipótesis conjunta de Fama
(1970) — no hace falta un modelo de equilibrio de activos para saber si el precio
era correcto. Basta esperar al pitido final.

**Pregunta de investigación.** ¿Contienen las tendencias estadísticas de acceso
público (Linemate) información sobre córneres, tarjetas y hándicaps que las cuotas
de BetPlay no hayan incorporado, en magnitud suficiente para generar retornos
anormales netos del margen?

**Contribuciones declaradas:**

1. Evidencia sobre un mercado **poco estudiado**: micro-mercados (córneres,
   tarjetas, *props*) de una casa latinoamericana, frente a la literatura
   dominante centrada en 1X2 de casas europeas.
2. Un **protocolo de paper trading pre-registrado** con registro inmutable ex
   ante, replicable y auditable.
3. Un **hallazgo metodológico**: la especificación Logit habitual en esta
   literatura sufre sesgo de variable omitida y **rechaza la EMH cuando esta es
   cierta**. Se propone y valida la corrección.

**Adelanto de resultados y hoja de ruta.**

> *Nota de honestidad intelectual:* la introducción debe adelantar el resultado
> obtenido, sea cual sea. Si no se detecta alfa, el paper es una **confirmación
> de la EMH en micro-mercados**, resultado valioso y publicable. No hay que
> "vender" una ineficiencia que los datos no respalden.

## 2. Marco Teórico (~2.000 palabras)

### 2.1 La EMH aplicada a mercados de apuestas
- Fama (1970): formas débil, semifuerte y fuerte. Este estudio contrasta la
  **semifuerte** (información pública).
- La condición de eficiencia con margen: el mercado es eficiente si
  $\mathbb{E}[\text{PnL}] = -m/(1+m)$ para toda estrategia basada en información
  pública. Superar ese umbral es la definición operativa de alfa.
- Sauer (1998), *The Economics of Wagering Markets*: revisión de referencia.
- Thaler & Ziemba (1988): por qué estos mercados **deberían** ser eficientes
  (retroalimentación rápida, incentivos claros, activos de vida corta).

### 2.2 Anomalías documentadas
- **Sesgo favorito-longshot** (Griffith, 1949; Snowberg & Wolfers, 2010): los
  *longshots* están sistemáticamente sobrevalorados. Justifica el uso de Shin
  frente a la normalización proporcional y el filtro `odds_max`.
- **Sesgo de equipo popular** (Forrest & Simmons, 2008): el volumen sesga la línea.
- **Subreacción a información reciente** (analogía con Bernard & Thomas, 1989):
  el parámetro $\lambda$ del DGP formaliza exactamente esta hipótesis.

### 2.3 Límites al arbitraje
Sección crítica: **aunque exista una ineficiencia, puede no ser explotable.**
- Shleifer & Vishny (1997): el arbitraje real es limitado y arriesgado.
- Restricciones específicas del mercado de apuestas:
  - **Limitación de cuentas** (*limiting*): la casa recorta el stake máximo del
    apostador rentable. Es el límite al arbitraje más severo y explica por qué
    una ineficiencia puede persistir en equilibrio.
  - Imposibilidad de posición corta sobre una selección concreta.
  - Capital inmovilizado y riesgo de ruina.
- **Implicación teórica:** la EMH puede fallar estadísticamente y sostenerse
  económicamente. Esta distinción debe estructurar la discusión.

### 2.4 Asimetría de información en micro-mercados
- Shin (1993): el margen como compensación de la casa frente a informados. El
  parámetro $z$ estimado por el de-vigging es una **medida directa** de la
  asimetría percibida.
- **Hipótesis central del trabajo:** los micro-mercados (córneres, tarjetas,
  *props*) reciben menos volumen y menos atención de apostadores sofisticados que
  el 1X2, luego su precio debería ser **menos eficiente** y su margen **mayor**.
  Ambas cosas actúan en direcciones opuestas sobre la rentabilidad: más
  ineficiencia, pero más impuesto por transacción. **Cuál domina es una pregunta
  empírica** — y es la pregunta de este paper.

### 2.5 La línea de cierre como referencia de eficiencia
Bassett (1981); Woodland & Woodland (1994). Si la línea de cierre agrega toda la
información disponible, batirla es condición necesaria de habilidad predictiva y
observarla es mucho menos ruidoso que observar resultados.

## 3. Metodología (~2.500 palabras)

### 3.1 Datos
Fuentes, periodo, mercados cubiertos, frecuencia de captura, protocolo de
emparejamiento entre fuentes, tratamiento de faltantes.
**Incluir consideraciones éticas y legales del scraping** (`robots.txt`, *rate
limiting*, uso académico, fecha de consulta de los Términos de Servicio).

### 3.2 Construcción de probabilidades
1. Implícita bruta: $q_i = 1/c_i$
2. Overround: $m = \sum_i q_i - 1$
3. Neutralización (cuatro estimadores; Shin principal)
4. Estimación propia $\hat{p}$: encogimiento empírico-bayesiano + modelo de
   conteo, combinados por pool logarítmico
5. Filtro: $\text{EV} = \hat{p}c - 1 > \tau$

**Subsección obligatoria — Identificación.** Explicar por qué $\hat{p}$ no puede
derivarse del propio libro de BetPlay (EV idénticamente $-m/(1+m)$), y por qué el
umbral $\tau > 0$ es necesario para absorber el error de estimación y evitar la
selección adversa (*maldición del ganador* aplicada a la selección de apuestas).

### 3.3 Diseño de paper trading
Stake plano de 1 unidad; registro ex ante inmutable; límite de apuestas por
partido; regla de parada por calendario; archivo de las decisiones rechazadas.

### 3.4 Estrategia empírica
Contrastes H1–H4 (ver `docs/PREREGISTRO.md`), tratamiento de la dependencia
(agrupamiento, HAC, bootstrap por bloques), corrección por multiplicidad.

### 3.5 Potencia declarada ex ante
**Sección diferenciadora.** Casi ningún trabajo de esta literatura declara su
potencia. Con $\sigma \approx 1$ por apuesta, detectar un ROI del 3% exige
$n \ge 6.870$; una temporada rinde 600–800 observaciones. Declararlo por
adelantado transforma un fallo del diseño en una limitación gestionada y
justifica el papel central del CLV.

## 4. Resultados Econométricos (~2.000 palabras)

**Tabla 1.** Descriptivos: $n$, cuota media, margen medio por mercado, EV medio ex
ante, tasa de acierto, ROI.
*Comentar el margen por mercado:* si los micro-mercados cobran un vig mucho mayor
que el 1X2, esa es ya una evidencia de partida sobre dónde está el impuesto.

**Tabla 2.** Contrastes sobre $\mu_{ROI}$: $t$, HAC, agrupado, bootstrap. Reportar
las cuatro columnas; la coincidencia entre ellas es la prueba de robustez.

**Tabla 3.** CLV: contraste principal por potencia.
*Interpretación clave:* un CLV positivo con ROI no significativo indica **señal
real ahogada por el ruido muestral**; un CLV negativo con ROI positivo indica
**suerte**, y debe decirse con esas palabras.

**Tabla 4.** Regresión de eficiencia (especificación principal):
$$\ln\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 \ln\left(\frac{p_i^{mkt}}{1-p_i^{mkt}}\right) + \beta_2 \text{TrendScore}_i + \epsilon_i$$
Contrastar $\beta_1 = 1$ (calibración) y $\beta_2 = 0$ (información no incorporada).

**Tabla 5.** Especificación con EV y CLV; VIF; efectos marginales.
**Discutir explícitamente la colinealidad** entre EV y TrendScore: es estructural,
no accidental, porque EV se construye a partir del TrendScore. Reportar el Wald
conjunto además de los coeficientes individuales.

**Tabla 6 — Apéndice metodológico.** Comparación de la especificación ingenua
frente a la corregida sobre datos simulados bajo eficiencia perfecta, mostrando el
falso positivo. **Es la contribución metodológica del paper: dedicarle espacio.**

**Figuras.** (1) Evolución del bankroll con banda de confianza bootstrap;
(2) curva de calibración predicho-observado; (3) ROI y CLV por mercado;
(4) convergencia de la línea (apertura → cierre).

**Robustez.** Cuatro métodos de de-vigging; exclusión de *props*; submuestras por
tramo de cuota; Kelly fraccionario frente a stake plano; exclusión de las jornadas
iniciales (calentamiento del modelo).

## 5. Discusión y Conclusiones (~1.500 palabras)

### 5.1 Interpretación según el resultado obtenido

- **Si no hay alfa (escenario más probable):** la EMH semifuerte no se rechaza en
  micro-mercados de LaLiga. Las tendencias de Linemate son **información pública
  ya incorporada al precio**, y el margen del 5–10% actúa como barrera
  infranqueable para una señal de esta relación señal-ruido. Enfatizar que una
  tendencia "8 de 10" tiene un error estándar de 0,13: es **estadísticamente casi
  indistinguible** de la probabilidad de mercado.

- **Si hay alfa:** antes de proclamar ineficiencia, agotar las explicaciones
  alternativas — (i) suerte, contrastada con el CLV; (ii) sesgo de selección de
  mercados; (iii) *data snooping*, mitigado por el pre-registro; (iv) sesgo de
  supervivencia de la fuente de tendencias. Solo si sobreviven todas, discutir si
  la ineficiencia es **económicamente explotable** dados los límites al arbitraje.

### 5.2 El puente conductual
Si la señal predice pero el precio no la incorpora, conectar con la subreacción a
información reciente y con la falacia del tamaño muestral pequeño (Tversky &
Kahneman, 1971): tanto el apostador que sobrerreacciona a "8 de 10" como la casa
que no ajusta lo suficiente son manifestaciones del mismo sesgo cognitivo, en
direcciones opuestas.

### 5.3 Limitaciones
Potencia; una sola casa y una sola liga; ausencia de *limiting* y deslizamiento en
el paper trading; estabilidad de los extractores; horizonte de una temporada.

### 5.4 Investigación futura
Extensión multi-casa (detección de arbitraje y de líneas atípicas); modelos
Poisson bivariantes con dependencia (Dixon & Coles, 1997); mercados en vivo;
medición directa del *limiting* como límite al arbitraje.

---

## Consejos para la defensa universitaria

**Las cinco preguntas que le harán, y su respuesta:**

1. *"¿Y si simplemente tuvo suerte?"* → Por eso el CLV es el contraste principal y
   por eso la potencia se declaró ex ante. Muestre el análisis de potencia.
2. *"¿Por qué su EV no es circular?"* → Sección 3.2, subsección de identificación.
   $\hat{p}$ proviene de un conjunto de información distinto del precio; si no
   fuera así, el EV sería $-m/(1+m)$ idénticamente, y lo tiene demostrado y
   verificado en el test correspondiente.
3. *"¿Por qué Shin y no la normalización simple?"* → Sesgo favorito-longshot; y
   $z$ tiene interpretación económica. Muestre la tabla de robustez con los cuatro
   métodos.
4. *"Su muestra es pequeña."* → Reconózcalo de entrada: está pre-registrado como
   limitación y por eso la inferencia principal descansa en el CLV. Convertir una
   debilidad anticipada en una decisión de diseño es lo que distingue un trabajo
   riguroso.
5. *"¿Esto significa que se puede ganar dinero?"* → Distinga **ineficiencia
   estadística** de **explotabilidad económica** (límites al arbitraje, *limiting*).
   Es la distinción que demuestra madurez económica.

**Errores a evitar:**
- Presentar un ROI positivo no significativo como si fuera un hallazgo.
- Omitir el margen al calcular rentabilidad.
- Usar la especificación Logit sin control de precio (rechaza la EMH siendo cierta).
- Cambiar los parámetros del modelo tras ver los resultados.

## Referencias mínimas

Angrist & Pischke (2009) · Bassett (1981) · Bernard & Thomas (1989) · Dixon &
Coles (1997) · Fama (1970) · Forrest & Simmons (2008) · Griffith (1949) · Sauer
(1998) · Shin (1993) · Shleifer & Vishny (1997) · Snowberg & Wolfers (2010) ·
Thaler & Ziemba (1988) · Tversky & Kahneman (1971) · Woodland & Woodland (1994)
