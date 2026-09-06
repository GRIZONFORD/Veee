# Nota Metodológica

## 1. El problema de identificación

El filtro de valor esperado es

$$\text{EV}_i = \hat{p}_i \cdot c_i - 1$$

Si $\hat{p}$ se obtiene neutralizando el margen de **la misma casa** sobre la que
se apuesta, entonces, por construcción,

$$\sum_i p_i^{\text{devig}} c_i - 1 = \frac{1}{\sum_j 1/c_j} \sum_i \frac{c_i}{c_i} - 1 = \frac{n}{1+m} - 1$$

y para cada selección individual $p_i^{\text{devig}} c_i - 1 = -m/(1+m) < 0$
**uniformemente**. Es decir: apostar contra el precio neutralizado de la propia
casa genera un EV negativo constante e igual al margen. El filtro $+$EV sería
vacuo.

> **Hipótesis de identificación:** $\hat{p}$ debe construirse con un conjunto de
> información distinto del precio de BetPlay. En este diseño: las tendencias de
> Linemate y un modelo de conteo Poisson/Binomial Negativa.

Verificado en `tests/test_oddsmath.py::test_devig_no_genera_ev_en_su_propio_libro`.

## 2. Elección del estimador de de-vigging

| Método | Supuesto | Sesgo conocido |
|--------|----------|----------------|
| Proporcional | Margen proporcional a $q_i$ | Sobreestima *longshots* |
| Aditivo | Margen repartido por igual | Degenera con cuotas dispares |
| Potencia | $p_i = q_i^k$ | Corrige parcialmente el sesgo F-L |
| **Shin** | Margen = defensa frente a informados | Preferido; $z$ es interpretable |

Se adopta **Shin** como especificación principal por dos razones: corrige el sesgo
favorito-longshot y su parámetro $z$ tiene contenido económico directo — es una
medida de la asimetría de información que la casa percibe, magnitud central para
el marco teórico. Los otros tres se reportan como robustez.

## 3. El sesgo de variable omitida en el Logit

**Este es el hallazgo metodológico más importante del diseño.**

La especificación habitual

$$\ln\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 \text{EV}_i + \beta_2 \text{TrendScore}_i + \beta_3 \text{CLV}_i + \epsilon_i$$

está **mal especificada**: omite el precio de mercado. La probabilidad de acierto
depende mecánicamente del nivel de la cuota (una apuesta a 5.00 gana menos veces
que una a 1.50), y el TrendScore está correlacionado con la cuota por el propio
proceso de selección — solo se apuesta cuando la señal discrepa del precio.

En simulación sobre un mercado **perfectamente eficiente** (`escenario_h0`), esta
especificación arroja $\hat\beta_2 > 0$ con $p < 0.001$: **rechazaría la EMH
siendo esta cierta**. Documentado en
`tests/test_econometrics.py::test_especificacion_ingenua_rechaza_emh_siendo_cierta`.

### Especificación correcta

$$\ln\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 \ln\left(\frac{p_i^{\text{mercado}}}{1-p_i^{\text{mercado}}}\right) + \beta_2 \text{TrendScore}_i + \epsilon_i$$

Es la regresión de eficiencia al estilo Fama. Bajo la EMH semifuerte:

$$H_0:\ \beta_0 = 0,\quad \beta_1 = 1,\quad \beta_2 = 0$$

- $\beta_1 = 1$: el precio está perfectamente calibrado.
- $\beta_1 < 1$: **atenuación** — el precio es demasiado extremo o demasiado
  ruidoso frente al resultado real. Firma típica del ruido de fijación de precios.
- $\beta_2 \neq 0$: información pública no incorporada al precio → ineficiencia.

Validado: bajo H0 no rechaza ($p=0.20$); bajo ruido de precio detecta
$\beta_1 = 0.69$ con $p = 0.001$.

## 4. Endogeneidad del CLV

El CLV se realiza **después** de colocar la apuesta. Incluirlo como regresor lo
convierte en un *bad control* (Angrist & Pischke, 2009, cap. 3): es un mediador
entre la señal y el resultado, de modo que condicionar sobre él sesga $\beta_2$.
Se estiman por separado:

- **Especificación ex ante** (sin CLV): efecto total de la señal. Es la que
  responde a la pregunta de investigación.
- **Especificación con CLV**: descomposición mediador/directo. Interpretación
  descriptiva únicamente.

## 5. Por qué el CLV es el contraste de mayor potencia

La varianza del retorno de una apuesta binaria es

$$\sigma = c\sqrt{p(1-p)}$$

que alcanza $\sigma = 1$ en $c = 2.00$. Detectar un ROI del 3% exige

$$n \ge \left(\frac{(z_{1-\alpha}+z_{1-\beta})\sigma}{\text{ROI}}\right)^2 \approx 6.870 \text{ apuestas.}$$

El CLV, en cambio, se mide **sin el ruido del resultado aleatorio**: compara dos
precios, no un precio contra una realización Bernoulli. En simulación, con la
misma muestra, el estadístico $t$ del CLV supera al del ROI en un orden de
magnitud ($t \approx 11$ frente a $t \approx -0.16$).

Fundamento teórico: si la línea de cierre es el estimador más eficiente de la
probabilidad real (Bassett, 1981; Woodland & Woodland, 1994), batirla
sistemáticamente es evidencia de captura de información, mientras que un ROI
positivo en muestra pequeña es indistinguible de la suerte.

## 6. Dependencia y errores estándar

Tres fuentes de dependencia, tratadas explícitamente:

1. **Intra-partido**: varias apuestas del mismo encuentro comparten choques
   (expulsión, criterio arbitral) → errores estándar agrupados por `match_id`.
2. **Serial**: sesgos persistentes del modelo entre jornadas → HAC Newey-West.
3. **No normalidad**: el PnL es fuertemente asimétrico → bootstrap por bloques
   con $H_0$ impuesta por centrado.

## 7. Limitaciones reconocidas

- **Paper trading**: no incorpora limitación de apuestas (*limiting*), rechazo de
  apuestas ni deslizamiento de precio. En la práctica, una estrategia rentable
  detectada por la casa ve su stake máximo recortado — una forma concreta de
  *limits to arbitrage*.
- **Sin costes de transacción más allá del margen**: se omiten tiempo de
  búsqueda, coste de oportunidad del capital inmovilizado y tratamiento fiscal.
- **Validez externa**: los resultados aplican a LaLiga en BetPlay Colombia, en una
  temporada. No se extrapolan a otras ligas, casas o periodos.
- **Estabilidad de los extractores**: los selectores del DOM no son estables; una
  ruptura silenciosa introduciría datos faltantes no aleatorios.
