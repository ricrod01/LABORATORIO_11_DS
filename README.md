# Laboratorio 11: MLOps del Observatorio Turistico

El repositorio ejecuta un flujo reproducible de lectura, controles de calidad,
evaluacion temporal, pronostico, monitoreo y publicacion controlada. El tablero
solo consulta la version indicada por `resultados/version_vigente.json`.

## Requisitos

- Docker Desktop con Docker Compose v2.
- Git, para registrar el commit de la ejecucion.
- La fuente institucional en `datos/DATOS_ORIGINALES.csv`.

Los datos institucionales no se copian a la imagen. `compose.yaml` los monta en
tiempo de ejecucion y el pipeline nunca modifica `DATOS_ORIGINALES.csv`.

## Configuracion inicial

1. Revise `scripts/config.yaml`, principalmente rutas, periodos, vias, umbrales
   de aceptacion y el umbral de variacion de monitoreo.
2. Coloque la fuente autorizada en `datos/DATOS_ORIGINALES.csv`.
3. Conserve exactamente los encabezados definidos en `limpieza.columnas`.
4. Defina el commit para que tambien quede disponible dentro de la imagen.

PowerShell:

```powershell
$env:GIT_COMMIT = git rev-parse HEAD
docker compose build
```

Bash:

```bash
export GIT_COMMIT="$(git rev-parse HEAD)"
docker compose build
```

## Inicio y acceso

Inicie MLflow y el tablero:

```bash
docker compose up -d mlflow dashboard
docker compose ps
```

- Tablero: <http://localhost:8501>
- MLflow Tracking: <http://localhost:5000>

El servicio `pipeline` usa un perfil para que levantar el tablero no vuelva a
ajustar modelos. Los filtros de Streamlit solo leen archivos publicados.

## Ejecutar el pipeline

```bash
docker compose run --rm pipeline
```

El comando devuelve un codigo distinto de cero si falla un control o la version
es rechazada. En ese caso no cambia `resultados/version_vigente.json`.

Ejecucion local alternativa, con MLflow ya disponible:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python .\scripts\pipeline.py --config .\scripts\config.yaml
```

## Monitoreo

Cada ejecucion deja:

- `logs/pipeline_<ID>.log`: inicio, fin, etapas, advertencias y errores.
- `reportes/<ID>/ejecucion.json`: estado completo y resultado de publicacion.
- `reportes/<ID>/monitoreo.json`: verificaciones de actualizacion, calidad y resultados.
- MLflow: parametros, metricas, controles, modelos y artefactos de evidencia.

Las verificaciones de monitoreo son:

- **Actualizacion:** compara la fuente con `controles.final.cobertura.fecha_fin`
  y las salidas con el ultimo mes del horizonte configurado.
- **Calidad:** compara faltantes, duplicados y cobertura con los umbrales y con
  la version vigente.
- **Resultados:** detecta negativos/no finitos y variaciones mensuales o contra
  la version vigente mayores a `monitoreo.variacion_relativa_maxima`.

El umbral actual de variacion es 50%. Se utiliza como alerta de revision porque
el turismo puede tener estacionalidad pronunciada; una alerta no rechaza por si
sola una version ni se interpreta como perdida de precision. El error predictivo
solo se registra para los meses de prueba que tienen observaciones comparables.

## Prueba controlada de fallo

`datos/DATOS_FALLO.csv` omite la columna indispensable `Viajero`. La prueba
ejecuta el pipeline, exige que falle y comprueba mediante hashes que el puntero y
las estimaciones aprobadas no cambiaron:

```bash
docker compose run --rm pipeline python /app/scripts/prueba_fallo_controlado.py
```

La evidencia se guarda en
`reportes/pruebas_controladas/fallo_<ID>.json`. Después puede corregirse la
fuente y ejecutarse nuevamente `docker compose run --rm pipeline`.

## Versiones y restauracion

Las versiones aprobadas se conservan en `resultados/aprobados/<ID>/`. Liste las
disponibles y restaure una anterior sin eliminar las posteriores:

```bash
docker compose run --rm pipeline python /app/scripts/pipeline.py \
  --config /app/scripts/config.yaml --restaurar 20261031T123304Z_888cb1b5
```

La restauracion verifica el hash, la aprobacion, el esquema de estimaciones y la
presencia de los modelos. Registra evidencia en `reportes/restauraciones/` y
cambia atomicamente el puntero que consulta el tablero. Recargue
<http://localhost:8501> para consultar la version restaurada.

Para volver a la version posterior use el mismo comando con su identificador.

## Reproducibilidad

Comparacion de dos ejecuciones equivalentes:

```bash
docker compose run --rm pipeline python /app/scripts/pipeline.py \
  --config /app/scripts/config.yaml --comparar ID_1 ID_2
```

La configuracion ejecutada, versiones efectivas de dependencias, commit, huellas
del codigo y fuente quedan en el reporte y en MLflow. Los rangos compatibles de
dependencias estan declarados en `requirements.txt` y la imagen fija Python 3.12.

## Persistencia

- MLflow usa el volumen nombrado `mlflow-data` para SQLite y artefactos.
- Datos, resultados, modelos, reportes y logs son carpetas montadas del host.
- `docker compose down` elimina contenedores y red, pero conserva ambos tipos de
  almacenamiento.
- No use `docker compose down -v` salvo que quiera borrar los registros nuevos
  de MLflow.

La base histórica `mlflow.db` del repositorio contiene rutas absolutas de otra
computadora y no se usa dentro de Docker. El servicio crea un backend portable
en `mlflow-data`; las ejecuciones nuevas persisten al recrear contenedores.

## Detencion

```bash
docker compose down
```

## Copia limpia

Desde un clon nuevo solo se necesita incorporar la fuente institucional y luego
ejecutar:

```bash
git clone <URL_PRIVADA>
cd LABORATORIO_11_DS
export GIT_COMMIT="$(git rev-parse HEAD)"
docker compose build
docker compose up -d mlflow dashboard
docker compose run --rm pipeline
```

No se requieren archivos intermedios de una sesion previa. Si aun no hay una
version aprobada, el tablero informa que las estimaciones no estan disponibles.
