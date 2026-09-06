# Guía de Despliegue — Captura en Vivo

> **Estado actual: NO-GO.** `config/config.yaml` contiene marcadores de posición
> (`ENDPOINT.A.CALIBRAR`). El sistema está completo y probado, pero no puede
> capturar hasta que se calibren los extractores contra los sitios reales.
> `scripts/preflight.py` lo impide activamente.

## Por qué no se pudo lanzar desde la sesión de desarrollo

Tres bloqueos independientes, ninguno resoluble por reintento:

1. **Red denegada por política.** El entorno remoto deniega `betplay.com.co` y
   `linemate.io` (respuesta 403 del gateway al CONNECT). Verificado contra el
   estado del proxy; GitHub responde 200 desde el mismo entorno.
2. **Contenedor efímero.** La sesión se recicla por inactividad; una recolección
   de varias jornadas no puede vivir en ella.
3. **Endpoints desconocidos.** Ni la API interna de BetPlay ni la de Linemate
   están documentadas públicamente, y sin red no se pueden descubrir.

La captura debe ejecutarse en una máquina propia, preferiblemente **en Colombia**
(BetPlay es un operador con licencia local y puede aplicar bloqueo geográfico).

---

## Paso 1 — Calibrar los extractores

La forma más rápida es leer la API interna del sitio en lugar del DOM: es más
estable y mucho más ligera para el servidor.

```bash
# a) Abra el sitio en el navegador, pestaña Red, filtro XHR/Fetch.
#    Localice la petición que devuelve el árbol de eventos en JSON y copie su URL.

# b) Descárguela y examine su estructura
./scripts/calibrate.py fetch "https://<url-de-la-api>" --out data/raw/muestra.json
./scripts/calibrate.py tree data/raw/muestra.json --max-depth 4

# c) Traslade las rutas a config.yaml > betplay.json_paths
#    (events, home, away, start, offers)

# d) Si no hay API y hay que parsear el DOM:
./scripts/calibrate.py fetch "https://<url-html>" --out data/raw/pagina.html
./scripts/calibrate.py selectors data/raw/pagina.html   # propone selectores CSS

# e) Verifique que la configuración extrae de verdad
./scripts/calibrate.py verify --fuente betplay
./scripts/calibrate.py verify --fuente linemate
```

**Nota legal.** Revise los Términos de Servicio de cada plataforma y deje
constancia de la fecha de consulta en la sección de Metodología del paper. El
cliente respeta `robots.txt` y aplica *rate limiting*; no lo desactive.

## Paso 2 — Preflight (obligatorio)

```bash
./scripts/preflight.py
```

Valida configuración, DNS/TLS, `robots.txt`, contrato de datos y —lo más
importante— **que las claves de partido de ambas fuentes crucen entre sí**. Si la
canonización de nombres de equipo difiere, el cruce da cero, no se registra
ninguna apuesta y el sistema *parece* funcionar durante semanas. Los nombres
huérfanos se listan para añadirlos a `ALIAS_EQUIPOS` en
`src/veee/scrapers/betplay.py`.

No active el cron hasta obtener **GO**.

## Paso 3 — Congelar el pre-registro

Antes del primer tick, fije y feche `docs/PREREGISTRO.md`. Los parámetros del
modelo no deben tocarse una vez iniciada la recolección: cambiarlos después de
ver resultados invalida el contraste de hipótesis.

```bash
git add docs/PREREGISTRO.md config/config.yaml
git commit -m "pre-registro congelado antes de la recoleccion"
git tag preregistro-$(date -u +%F)
```

## Paso 4 — Activar la captura

### Opción A — cron (recomendada)

```bash
sudo useradd -r -m -d /opt/veee veee
sudo -u veee git clone <repo> /opt/veee && cd /opt/veee
sudo -u veee python3 -m venv .venv && sudo -u veee ./.venv/bin/pip install -r requirements.txt
sudo -u veee mkdir -p logs data
sudo -u veee crontab deploy/crontab.example
```

### Opción B — systemd (mejor: `Persistent=true` recupera ticks tras un apagón)

```bash
sudo cp deploy/veee-capture.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now veee-capture.timer
systemctl list-timers veee-capture.timer
```

### Opción C — GitHub Actions (`.github/workflows/captura.yml`)

Solo si el sitio no aplica bloqueo geográfico. Requiere el secreto
`VEEE_CONFIG_YAML` con la configuración calibrada. Dos limitaciones que afectan
al estudio:

- Los runners están fuera de Colombia → riesgo de bloqueo geográfico.
- El planificador se retrasa 5–15 min en horas de carga → **degrada la línea de
  cierre**, que es el dato de mayor valor. Aceptable como respaldo, no como
  mecanismo principal.

## Cómo funciona la captura

Escalera de ventanas anclada al pitido inicial, cada una capturada **una sola vez**:

```
T-72h  T-48h  T-24h  T-12h  T-6h  T-3h  T-1h  T-30m  T-10m  T-3m
  └──── apertura ────┴─── convergencia ───┴──── CIERRE ────┘
         [ ventana de colocación: T-24h → T-6h ]
```

- **Capturar ≠ apostar.** Se registran precios en toda la escalera, pero las
  apuestas solo dentro de la ventana pre-registrada. Si se apostara en cualquier
  instante, el CLV mediría en parte la elección del momento y no la calidad de la
  señal, y los partidos dejarían de ser comparables.
- **Línea de cierre.** Tras el pitido, `finalize_closing_lines()` toma el último
  snapshot *anterior* al inicio de cada mercado, lo marca `es_cierre=1`,
  neutraliza el margen y rellena `clv_odds` y `clv_prob`. Sin esto no hay CLV, y
  sin CLV el estudio pierde su contraste de mayor potencia.
- **Huecos visibles.** Si cron estuvo caído, los hitos rebasados se registran con
  `n_filas = -1`. Los datos faltantes deben verse en la muestra, no desaparecer.
- **Idempotencia.** Reejecutar un tick no duplica snapshots de ventana ni
  apuestas: los IDs son deterministas.

## Vigilancia

Un cambio de DOM no produce un error, produce **cero filas**: datos faltantes no
aleatorios que sesgarían la muestra sin dejar rastro. `health_check` compara cada
tick contra la mediana móvil de los 30 anteriores.

| Código de salida | Significado | Acción |
|---|---|---|
| 0 | Correcto | — |
| 1 | Error de ejecución | Revisar `logs/capture.err` |
| 2 | **Alerta**: volumen < 50% de la mediana | Revisar selectores |
| 3 | **Crítico**: cero filas | Extractor roto; recalibrar |

```bash
# Estado de salud reciente
sqlite3 data/veee.db "SELECT ts,fuente,n_filas,estado,detalle FROM health_log
                      ORDER BY ts DESC LIMIT 20;"

# Cobertura de la escalera (¿se está capturando el cierre?)
sqlite3 data/veee.db "SELECT ventana_h, COUNT(*) n, SUM(n_filas<0) perdidas
                      FROM capture_log GROUP BY ventana_h ORDER BY ventana_h DESC;"

# Apuestas aún sin CLV (deberían ser solo las de partidos no jugados)
sqlite3 data/veee.db "SELECT COUNT(*) FROM bets WHERE cuota_cierre IS NULL;"
```

Alerta por correo desde cron: `MAILTO=usted@ejemplo.com` en el crontab basta,
porque el script sale con código distinto de cero cuando hay problemas.

## Rutina semanal

```bash
./scripts/settle.py --resultados data/resultados_YYYY-MM-DD.json  # liquidar
./scripts/run_analysis.py                                          # informe
```

**No mire el ROI acumulado cada día.** La regla de parada es por calendario, no
por resultado: detenerse al ver un $p$-valor favorable infla arbitrariamente la
tasa de error de tipo I. Está en el pre-registro por una razón.

## Copias de seguridad

`data/veee.db` es el estudio entero. Si se pierde, no se puede reconstruir: los
precios históricos no son recuperables a posteriori.

```bash
0 3 * * * cd /opt/veee && sqlite3 data/veee.db ".backup 'backups/veee-$(date -u +\%F).db'"
```
