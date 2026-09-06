# Protocolo de Pre-registro

> Este documento debe **congelarse y fecharse antes** de registrar la primera
> apuesta. Su función es impedir el *data snooping*: sin él, cualquier resultado
> positivo es indistinguible de una búsqueda sobre especificaciones.

## 1. Hipótesis (declaradas ex ante)

| Id | Hipótesis nula | Alternativa | Contraste |
|----|----------------|-------------|-----------|
| **H1** | $\mu_{ROI} \le 0$ | $\mu_{ROI} > 0$ | $t$ unilateral sobre ROI diario + bootstrap por bloques |
| **H2** | $\mu_{CLV} \le 0$ | $\mu_{CLV} > 0$ | $t$ unilateral sobre CLV |
| **H3** | $\beta_{trend} = 0$ dado el precio | $\beta_{trend} \neq 0$ | Wald sobre la regresión de eficiencia |
| **H4** | $\beta_{precio} = 1$ | $\beta_{precio} \neq 1$ | Contraste de calibración del precio |

**Contraste primario:** H1. Los demás son confirmatorios/secundarios y se
reportan siempre, con independencia del signo del resultado.

## 2. Parámetros congelados

Fijados en la muestra de calibración (pretemporada) y **no modificables** una vez
iniciada la recolección. Cualquier cambio genera un estudio nuevo, no una revisión
del presente.

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `k_shrink` | 12.0 | Encogimiento equivalente a ~12 partidos de prior |
| `devig_method` | `shin` | Corrige el sesgo favorito-longshot del método proporcional |
| `ev_min` | 0.03 | Absorbe el error de estimación de $\hat{p}$ |
| `trend_z_min` | 1.0 | Exige señal de al menos 1 desviación estándar |
| `odds_min` / `odds_max` | 1.40 / 6.00 | Evita cuotas cortas ruidosas y *longshots* |
| `max_bets_per_match` | 2 | Acota la dependencia intra-evento |
| Stake | 1 unidad (plano) | El ROI es una media no ponderada |
| $\alpha$ | 0.05 | Unilateral |

## 3. Regla de parada

La recolección termina en una **fecha calendario prefijada** (fin de la primera
vuelta de LaLiga), **nunca** al alcanzar un resultado significativo. Detenerse al
ver un $p$-valor favorable infla la tasa de error de tipo I de forma arbitraria.

## 4. Potencia declarada

Con cuota media 2.00 ($\sigma \approx 1$), detectar un ROI del 3% con potencia
0.80 y $\alpha = 0.05$ requiere $n \ge 6.870$ apuestas. Una
temporada de LaLiga con 2 apuestas por partido rinde del orden de 600–800
observaciones.

**Consecuencia asumida ex ante:** el estudio está **infrapotenciado** para el
contraste de ROI. El efecto mínimo detectable con $n=800$ es un ROI del **8,8%**,
magnitud implausible en un mercado maduro. Por ello:

1. H2 (CLV) es el contraste de **mayor potencia** y se privilegia en la discusión.
2. Un ROI positivo no significativo **no** se interpretará como evidencia de
   ineficiencia.
3. Un ROI negativo **no** se interpretará como confirmación de la EMH: es también
   compatible con una señal débil ahogada por el margen.

## 5. Compromisos de transparencia

- Se registran **todas** las evaluaciones, incluidas las rechazadas por el filtro
  (`outputs/decisiones/`), para caracterizar la selección muestral.
- Los campos ex ante son inmutables por diseño (`settle_bet` los rechaza).
- Los contrastes por mercado se corrigen por Benjamini-Hochberg.
- Se publica la matriz econométrica completa como anexo replicable.
- Se reporta el resultado **sea cual sea su signo**. La no-detección de alfa es
  un hallazgo publicable y es, de hecho, el resultado esperado a priori.

## 6. Análisis no pre-registrados

Cualquier análisis adicional se etiquetará explícitamente como **exploratorio** y
sus $p$-valores se interpretarán como descriptivos, no confirmatorios.
