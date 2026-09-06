# Veee — Ineficiencias en Mercados de Apuestas Deportivas

Marco econométrico reproducible para contrastar la **Hipótesis de Mercados
Eficientes (EMH)** en el mercado de apuestas de LaLiga: ¿contienen las tendencias
públicas de Linemate (córneres, tarjetas, hándicaps, *props*) información que las
cuotas de BetPlay no hayan incorporado?

Metodología: *paper trading* con stake plano, registro ex ante inmutable e
inferencia estadística sobre la rentabilidad ajustada por el margen de la casa.

## Instalación

```bash
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml   # y calibrar los selectores
```

## Uso

```bash
# 1) Ciclo diario: extrae, evalúa +EV y registra apuestas EX ANTE
python scripts/run_daily.py --fecha 2026-09-12
python scripts/run_daily.py --dry-run              # offline, con fixtures

# 2) Liquidación ex post: línea de cierre, resultado y PnL
python scripts/settle.py --resultados data/resultados_2026-09-12.json

# 3) Inferencia econométrica completa
python scripts/run_analysis.py
python scripts/run_analysis.py --simular h1_ruido  # sobre datos sintéticos

# 4) Estudio Monte Carlo de tamaño y potencia (pre-registro)
python scripts/power_study.py --rep 200

# Pruebas
python -m pytest tests/ -q                          # 69 pruebas
```

## Estructura

```
src/veee/
  oddsmath.py      Cuotas, de-vigging (proporcional/aditivo/potencia/Shin), EV, CLV, Kelly
  model.py         Construcción de p_est y regla de decisión ex ante
  scrapers/        BetPlay (precios) y Linemate (tendencias); cliente HTTP responsable
  database.py      Esquema SQLite y matriz econométrica
  pipeline.py      Orquestación del ciclo diario (idempotente)
  settlement.py    Liquidación, hándicaps asiáticos, CLV
  econometrics.py  Contrastes de hipótesis, Logit, diagnósticos, potencia
  simulate.py      DGP sintético para validar el diseño
docs/              PREREGISTRO.md · METODOLOGIA.md
paper/             ESQUEMA_PAPER.md
```

## Fundamentos del diseño

### 1. Identificación
$\hat{p}$ **no puede** derivarse neutralizando el propio libro de BetPlay: en ese
caso el EV de toda selección es idénticamente $-m/(1+m)$ y el filtro es vacuo. La
probabilidad estimada proviene de un conjunto de información distinto — las
tendencias de Linemate y un modelo de conteo Poisson/Binomial Negativa.

### 2. El Logit habitual está mal especificado
La especificación `y ~ EV + TrendScore + CLV`, **sin control del precio de
mercado**, sufre sesgo de variable omitida. En simulación sobre un mercado
**perfectamente eficiente** arroja un TrendScore positivo con $p<0.001$:
rechazaría la EMH siendo esta cierta. La especificación válida condiciona sobre el
precio:

$$\ln\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 \ln\left(\frac{p_i^{mkt}}{1-p_i^{mkt}}\right) + \beta_2 \text{TrendScore}_i + \epsilon_i$$

Bajo la EMH semifuerte: $\beta_1 = 1$ y $\beta_2 = 0$.

### 3. El CLV es el contraste de mayor potencia
Detectar un ROI del 3% con $\sigma \approx 1$ por apuesta exige $n \ge 6.870$
observaciones; una temporada rinde 600–800. El CLV se mide sin el ruido del
resultado aleatorio: en simulación, con la misma muestra, $t_{CLV} \approx 11$
frente a $t_{ROI} \approx -0.16$.

### 4. Protección anti *look-ahead*
Los campos ex ante son inmutables: `settle_bet()` rechaza cualquier intento de
reescribirlos. Los IDs de apuesta son deterministas, de modo que el pipeline es
idempotente. Se archivan **todas** las evaluaciones, incluidas las rechazadas.

## Validación del diseño

El DGP sintético reproduce tres escenarios y el marco los distingue correctamente:

| Escenario | ROI | CLV medio | $\beta_{precio}$ | Conclusión |
|-----------|-----|-----------|------------------|------------|
| `escenario_h0` (eficiente) | $\approx -m/(1+m)$ | $\approx -m/(1+m)$ | $\approx 1$ | No rechaza |
| `escenario_h1_subreaccion` | según $\lambda$ | crece al caer $\lambda$ | $<1$ | Subreacción |
| `escenario_h1_ruido` | $>0$ | $>0$ | $<1$ | Precio disperso |

## Aviso

Extracción de datos con fines exclusivamente académicos: se respeta `robots.txt`,
se aplica *rate limiting* y se identifica el agente. Verifique los Términos de
Servicio de cada plataforma antes de la recolección. **No se coloca dinero real:
todo el estudio es simulación sobre papel.** Los selectores del DOM y los
*endpoints* del archivo de configuración son plantillas y deben calibrarse.
