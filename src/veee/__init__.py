"""
veee — Evaluacion econometrica de la Hipotesis de Mercados Eficientes (EMH)
en el mercado de apuestas de LaLiga (BetPlay) frente a tendencias de Linemate.

Modulos
-------
oddsmath      Aritmetica de cuotas, de-vigging, EV, CLV, Kelly.
model         Construccion de p_est y regla de decision ex-ante.
scrapers      Extractores de BetPlay (precios) y Linemate (tendencias).
database      Esquema SQLite y matriz econometrica.
pipeline      Ciclo diario de recoleccion y registro ex-ante.
settlement    Liquidacion ex-post, linea de cierre y PnL.
econometrics  Contrastes de hipotesis, Logit y diagnosticos.
simulate      DGP sintetico para validacion del diseno y analisis de potencia.
"""
__version__ = "1.0.0"
__all__ = ["oddsmath", "model", "database", "pipeline", "settlement",
           "econometrics", "simulate", "config", "scrapers"]
