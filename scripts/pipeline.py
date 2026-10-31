"""Laboratorio 11 - orquestador reproducible, trazable y con publicación controlada.

Ejecutar desde el directorio del proyecto:
    python pipeline.py --config config.yaml
    python pipeline.py --config config.yaml --restaurar ID_APROBADO
    python pipeline.py --config config.yaml --comparar ID_1 ID_2

No modifica las fuentes institucionales y no publica resultados sin aprobación.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ajustes_modelos import (agrupa_via, ajuste_pandemia, ajustar_modelo,
                             entrenamiento_modelo, evaluar_modelo,
                             generacion_estimaciones, separacion_temporal)
from controles import (validacion_estimaciones, validacion_final,
                       validacion_inicial, validacion_metodologica)
from funciones_limpieza import (columnas_necesarias, columna_fecha,
                                configurar_logging, filtrado_filas,
                                guardar_datos, transformacion, consolidar_granularidad)


class FalloControl(RuntimeError):
    """Un control obligatorio no fue superado y se cancela la ejecución."""


def ahora():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def id_ejecucion():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]


def guardar_json(objeto, ruta):
    """Escritura atómica: un fallo no deja un JSON incompleto."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=ruta.parent,
                                         prefix='.temporal_', suffix='.json',
                                         delete=False) as archivo:
            temporal = Path(archivo.name)
            json.dump(objeto, archivo, indent=2, ensure_ascii=False, default=str)
            archivo.write('\n')
        os.replace(temporal, ruta)
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()
    return ruta


def hash_archivo(ruta):
    digest = hashlib.sha256()
    with open(ruta, 'rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def git_commit(carpeta):
    try:
        return subprocess.check_output(
            ['git', '-C', str(carpeta), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_modificaciones(carpeta):
    try:
        return subprocess.check_output(
            ['git', '-C', str(carpeta), 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def huellas_codigo(carpeta):
    nombres = ['pipeline.py', 'funciones_limpieza.py', 'controles.py', 'ajustes_modelos.py']
    return {nombre: hash_archivo(Path(carpeta) / nombre) for nombre in nombres
            if (Path(carpeta) / nombre).is_file()}


def leer_config(ruta_config):
    ruta_config = Path(ruta_config).resolve()
    with ruta_config.open('r', encoding='utf-8') as archivo:
        config = yaml.safe_load(archivo)
    if not isinstance(config, dict):
        raise ValueError('El archivo YAML no contiene una configuración válida')
    base = ruta_config.parent
    # Las rutas son relativas a config.yaml, nunca a la carpeta de trabajo actual.
    for clave, valor in config['rutas'].items():
        config['rutas'][clave] = str((base / valor).resolve())
    return config, ruta_config


def importar_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError as exc:
        raise RuntimeError('Debes instalar MLflow: pip install -r requirements.txt') from exc


def registrar_control(nombre, resultado, informe, mlflow, logger):
    informe['controles'][nombre] = resultado
    mlflow.log_metric(f'control_{nombre}', int(resultado['aprobado']))
    for advertencia in resultado['advertencias']:
        informe['advertencias'].append(f'{nombre}: {advertencia}')
        logger.warning('%s | %s', nombre, advertencia)
    for error in resultado['errores']:
        informe['errores'].append(f'{nombre}: {error}')
        logger.error('%s | %s', nombre, error)
    if not resultado['aprobado']:
        raise FalloControl(f'Control crítico no aprobado: {nombre}')
    logger.info('Control aprobado: %s', nombre)


def etapa(informe, logger, nombre, funcion):
    """Mide la duración de una etapa y registra incluso los intentos fallidos."""
    registro = {'nombre': nombre, 'inicio': ahora(), 'estado': 'ejecutando'}
    informe['etapas'].append(registro)
    logger.info('Etapa iniciada: %s', nombre)
    try:
        resultado = funcion()
        registro['estado'] = 'completada'
        return resultado
    except Exception:
        registro['estado'] = 'fallida'
        raise
    finally:
        registro['fin'] = ahora()
        logger.info('Etapa terminada: %s (%s)', nombre, registro['estado'])


def informacion_dependencias():
    paquetes = ['pandas', 'numpy', 'scikit-learn', 'statsmodels', 'prophet',
                'sktime', 'mlflow', 'PyYAML', 'joblib']
    resultado = {'python': platform.python_version(), 'sistema': platform.platform()}
    for paquete in paquetes:
        try:
            resultado[paquete] = version(paquete)
        except PackageNotFoundError:
            resultado[paquete] = 'no_instalado'
    return resultado


def preparar_carpeta_y_reporte(config, identificador):
    carpeta = Path(config['rutas']['reportes']) / identificador
    carpeta.mkdir(parents=True, exist_ok=False)
    informe = {
        'id_ejecucion': identificador, 'mlflow_run_id': None, 'inicio': ahora(),
        'fin': None, 'estado': 'ejecutando', 'publicacion': 'no_publicado',
        'etapas': [], 'controles': {}, 'advertencias': [], 'errores': [],
        'metricas': {}, 'modelos': {}, 'monitoreo': {}, 'origen': {},
        'archivos': {}, 'aceptacion': {}
    }
    return carpeta, informe


def leer_fuente(config, informe, mlflow):
    ruta = Path(config['rutas']['datos_originales'])
    if not ruta.is_file():
        raise FileNotFoundError(f'No se encuentra la fuente institucional: {ruta}')
    opciones = config.get('lectura', {})
    datos = pd.read_csv(
        ruta, sep=opciones.get('separador', ','),
        decimal=opciones.get('decimal', '.'),
        encoding=opciones.get('encoding', 'utf-8-sig'), low_memory=False
    )
    informe['origen'] = {'ruta': str(ruta), 'sha256': hash_archivo(ruta),
                         'registros': len(datos), 'frecuencia_declarada': 'mensual'}
    mlflow.log_param('fuente_sha256', informe['origen']['sha256'])
    mlflow.log_param('fuente_registros', len(datos))
    return datos


def limpiar_y_validar(datos, config, informe, mlflow, logger, carpeta):
    registrar_control('inicial', validacion_inicial(datos, config), informe, mlflow, logger)
    limpieza = config['limpieza']
    limpio = columnas_necesarias(datos, limpieza['columnas'])
    limpio = transformacion(limpio)
    limpio = filtrado_filas(limpio, limpieza['tipos_viajero'], limpieza['paises_excluidos'],
                           limpieza.get('vias_permitidas', config['modelos']['vias']))
    limpio = columna_fecha(limpio)
    limpio, detalle = consolidar_granularidad(limpio, config['controles']['final']['clave_unicidad'])
    informe['consolidacion'] = detalle
    guardar_json(detalle, carpeta / 'consolidacion.json')
    mlflow.log_metric('filas_consolidadas', detalle['filas_adicionales_consolidadas'])
    if detalle['filas_adicionales_consolidadas']:
        aviso = ('Consolidación por frontera, país, tipo y mes: '
                 f"{detalle['filas_adicionales_consolidadas']} filas agrupadas sin perder su suma")
        informe['advertencias'].append(aviso)
        logger.warning(aviso)
    registrar_control('final', validacion_final(limpio, config), informe, mlflow, logger)
    metadatos_fuente = {'ruta': informe['origen']['ruta'],
                       'frecuencia': informe['origen']['frecuencia_declarada']}
    registrar_control('metodologia', validacion_metodologica(
        limpio, config, metadatos_fuente), informe, mlflow, logger)
    guardar_datos(limpio, config['rutas']['datos_limpios'])
    informe['archivos']['datos_limpios'] = config['rutas']['datos_limpios']
    mlflow.log_metric('datos_limpios_registros', len(limpio))
    guardar_json(informe['controles'], carpeta / 'controles_calidad.json')
    return limpio


def entrenar_evaluar_pronosticar(limpio, config, informe, mlflow, logger, carpeta):
    estimaciones = []
    meses_por_via = {}
    for via in config['modelos']['vias']:
        logger.info('Modelado de la vía: %s', via)
        serie = ajuste_pandemia(agrupa_via(limpio, via, config), config)
        train, val, test = separacion_temporal(serie, config)
        _, tipo, mae_validacion, hiperparametros, fallos = entrenamiento_modelo(train, val, config)
        # Reajuste permitido: test no se utiliza hasta evaluar.
        anterior = pd.concat([train, val])
        modelo_evaluacion = ajustar_modelo(anterior, tipo, hiperparametros, config)
        evaluacion = evaluar_modelo(modelo_evaluacion, tipo, anterior, test, config)
        # Modelo de producción: utiliza todo el historial disponible al generar meses futuros.
        modelo_produccion = ajustar_modelo(serie, tipo, hiperparametros, config)
        ultimo_mes = serie.index.max()
        inicio_esperado = config['modelos']['pronostico'].get('fecha_inicio')
        if inicio_esperado is not None and pd.Period(inicio_esperado, freq='M') != ultimo_mes + 1:
            raise ValueError(f'{via}: pronóstico configurado para {inicio_esperado}, '
                             f'pero el primer mes posible tras los datos es {ultimo_mes + 1}')
        pred = generacion_estimaciones(modelo_produccion, tipo, ultimo_mes, via, config)
        estimaciones.append(pred)
        meses_por_via[via] = {'train': [str(train.index.min()), str(train.index.max())],
                              'validacion': [str(val.index.min()), str(val.index.max())],
                              'prueba': [str(test.index.min()), str(test.index.max())],
                              'pronostico': [str(p) for p in pd.PeriodIndex(pred['fecha'], freq='M')]}
        clave = via.lower().replace(' ', '_')
        mlflow.log_param(f'modelo_{clave}', tipo)
        mlflow.log_param(f'hiperparametros_{clave}', json.dumps(hiperparametros, ensure_ascii=False))
        metricas = {'mae_validacion': mae_validacion,
                    'mae_modelo': evaluacion['mae_modelo'],
                    'mae_referencia': evaluacion['mae_referencia']}
        for nombre, valor in metricas.items():
            mlflow.log_metric(f'{nombre}_{clave}', float(valor))
            informe['metricas'][f'{nombre}_{clave}'] = float(valor)
        informe['modelos'][via] = {
            'tipo': tipo, 'hiperparametros': hiperparametros,
            'metricas': metricas, 'meses_evaluados': evaluacion['meses_evaluados'],
            'periodos': meses_por_via[via], 'fallos_candidatos': fallos
        }
        evaluacion['comparacion'].to_csv(carpeta / f'evaluacion_{clave}.csv',
                                         index_label='periodo', encoding='utf-8')
        # Los parámetros son suficientes para reproducir el procedimiento; el objeto
        # ajustado también se guarda cuando su serialización es posible.
        try:
            import joblib
            ruta_modelo = Path(config['rutas']['modelos']) / informe['id_ejecucion'] / f'{clave}.joblib'
            ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo_produccion, ruta_modelo)
            informe['archivos'][f'modelo_{clave}'] = str(ruta_modelo)
            mlflow.log_artifact(str(ruta_modelo), artifact_path='modelos')
        except Exception as exc:
            aviso = f'No se pudo serializar el modelo {via}: {type(exc).__name__}: {exc}'
            logger.warning(aviso)
            informe['advertencias'].append(aviso)
        for fallo in fallos:
            logger.warning('%s | candidato descartado: %s', via, fallo)
    informe['periodos_por_via'] = meses_por_via
    return pd.concat(estimaciones, ignore_index=True)


def añadir_metadatos(predicciones, config, informe):
    publicacion = config['publicacion']
    salida = predicciones.copy()
    salida['destino'] = publicacion.get('destino')
    salida['periodo_estimado'] = salida['fecha'].dt.strftime('%Y-%m')
    salida['valor'] = salida['estimacion']
    salida['unidad'] = publicacion['unidad']
    salida['run_id'] = informe['mlflow_run_id']
    salida['id_ejecucion'] = informe['id_ejecucion']
    salida['fecha_generacion'] = ahora()
    salida['version_procedimiento'] = publicacion['version_procedimiento']
    salida['tipo_valor'] = 'estimado'
    salida['modelo'] = salida['via'].map({via: datos['tipo'] for via, datos in informe['modelos'].items()})
    return salida


def leer_version_vigente(config):
    ruta = Path(config['rutas']['version_vigente'])
    if not ruta.exists():
        return None
    with ruta.open('r', encoding='utf-8') as archivo:
        version = json.load(archivo)
    return version


def monitorear(predicciones, limpio, config, informe):
    alertas = []
    umbrales = config['monitoreo']
    cobertura = config['controles']['final']['cobertura']
    fin_esperado = cobertura.get('fecha_fin')
    fin_observado = limpio['fecha'].max().to_period('M')
    if fin_esperado is None:
        alertas.append('No se configuró fecha_fin de cobertura: no es posible verificar actualización contra el período esperado')
    elif fin_observado < pd.Period(fin_esperado, freq='M'):
        alertas.append(f'Fuente desactualizada: último período {fin_observado}; se esperaba {fin_esperado}')

    final = informe['controles']['final']['detalle']
    duplicados = int(final.get('duplicados', 0))
    nulos = sum(v for k, v in final.items() if k.startswith('nulos_') and isinstance(v, int))
    if duplicados > umbrales['umbral_duplicados']:
        alertas.append(f'Calidad: duplicados {duplicados} exceden el umbral')
    if nulos > umbrales['umbral_faltantes_criticos']:
        alertas.append(f'Calidad: valores faltantes {nulos} exceden el umbral')
    variacion = umbrales.get('variacion_relativa_maxima')
    if variacion is None:
        alertas.append('No se configuró variacion_relativa_maxima; monitoreo de cambios deshabilitado')
    else:
        for via, grupo in predicciones.sort_values('fecha').groupby('via'):
            valores = grupo['estimacion'].astype(float).to_numpy()
            if len(valores) > 1:
                cambios = np.abs(np.diff(valores)) / np.maximum(np.abs(valores[:-1]), 1)
                if np.any(cambios > variacion):
                    alertas.append(f'{via}: cambio mensual de {float(cambios.max()):.2%} supera {variacion:.2%}')
    anterior = leer_version_vigente(config)
    if anterior:
        metadatos = Path(anterior['ruta_metadatos'])
        if metadatos.is_file():
            with metadatos.open(encoding='utf-8') as archivo:
                anterior_info = json.load(archivo)
            anterior_calidad = anterior_info.get('calidad', {})
            if duplicados > anterior_calidad.get('duplicados', duplicados):
                alertas.append('Calidad: aumentaron los duplicados frente a la versión vigente')
            if nulos > anterior_calidad.get('nulos', nulos):
                alertas.append('Calidad: aumentaron los faltantes frente a la versión vigente')
            prev_cobertura = anterior_calidad.get('faltantes_cobertura', 0)
            faltan = sum(v.get('faltantes_criticos', 0) for k, v in final.items()
                         if k.startswith('cobertura_') and isinstance(v, dict))
            if faltan > prev_cobertura:
                alertas.append('Calidad: se redujo la cobertura frente a la versión vigente')
    informe['monitoreo'] = {'alertas': alertas, 'ultimo_mes_observado': str(fin_observado)}
    informe['advertencias'].extend(alertas)
    return {'duplicados': duplicados, 'nulos': nulos,
            'faltantes_cobertura': sum(v.get('faltantes_criticos', 0) for k, v in final.items()
                                      if k.startswith('cobertura_') and isinstance(v, dict))}


def decidir_aceptacion(config, informe, salida, control_salida):
    reglas = config['aceptacion']
    razones = []
    if reglas['exigir_controles_aprobados'] and any(not x['aprobado'] for x in informe['controles'].values()):
        razones.append('No se aprobaron todos los controles de calidad')
    if reglas['exigir_salidas_completas'] and not control_salida['aprobado']:
        razones.append('Las estimaciones no están completas o contienen valores inválidos')
    if reglas['exigir_todas_las_vias'] and set(salida['via']) != set(config['modelos']['vias']):
        razones.append('No hay estimaciones para todas las vías')
    if not config['publicacion'].get('destino'):
        razones.append('Debes configurar publicacion.destino')
    limite = reglas.get('mae_maximo')
    if limite is None or not np.isfinite(float(limite)) or float(limite) < 0:
        razones.append('Debes configurar un mae_maximo numérico y justificado')
    else:
        for via, info in informe['modelos'].items():
            mae = info['metricas']['mae_modelo']
            if mae > limite:
                razones.append(f'{via}: MAE {mae:.3f} supera el límite {limite:.3f}')
    if reglas['requerir_superar_referencia']:
        for via, info in informe['modelos'].items():
            if info['metricas']['mae_modelo'] >= info['metricas']['mae_referencia']:
                razones.append(f'{via}: el modelo no supera a la referencia')
    return {'aprobado': not razones, 'razones': razones,
            'criterios': reglas.copy(), 'evaluado_en': ahora()}


def auditar_publicacion(config, evento):
    # Registro independiente para evidenciar publicaciones y restauraciones.
    carpeta = Path(config['rutas']['reportes']) / 'historial_publicaciones'
    nombre = id_ejecucion() + '.json'
    guardar_json({'fecha': ahora(), **evento}, carpeta / nombre)


def publicar_version(config, informe, ruta_candidata, calidad, mlflow, logger):
    """Solo se cambia el puntero vigente cuando los archivos aprobados ya existen."""
    destino = Path(config['rutas']['resultados_aprobados'])
    destino.mkdir(parents=True, exist_ok=True)
    carpeta_final = destino / informe['id_ejecucion']
    carpeta_temporal = destino / f'.{informe["id_ejecucion"]}.temporal'
    if carpeta_final.exists() or carpeta_temporal.exists():
        raise FileExistsError('La versión aprobada ya existe')
    carpeta_temporal.mkdir()
    try:
        archivo_final = carpeta_temporal / 'estimaciones.csv'
        shutil.copy2(ruta_candidata, archivo_final)
        hash_salida = hash_archivo(archivo_final)
        metadatos = {
            'id_ejecucion': informe['id_ejecucion'], 'mlflow_run_id': informe['mlflow_run_id'],
            'fecha_publicacion': ahora(), 'sha256_estimaciones': hash_salida,
            'origen': informe['origen'], 'modelos': informe['modelos'],
            'calidad': calidad, 'aceptacion': informe['aceptacion'],
            'version_procedimiento': config['publicacion']['version_procedimiento']
        }
        guardar_json(metadatos, carpeta_temporal / 'metadatos.json')
        carpeta_temporal.replace(carpeta_final)
        puntero = {'id_ejecucion': informe['id_ejecucion'], 'mlflow_run_id': informe['mlflow_run_id'],
                   'ruta_estimaciones': str(carpeta_final / 'estimaciones.csv'),
                   'ruta_metadatos': str(carpeta_final / 'metadatos.json'),
                   'fecha_actualizacion': ahora(), 'accion': 'publicacion'}
        anterior = leer_version_vigente(config)
        guardar_json(puntero, config['rutas']['version_vigente'])
        try:
            auditar_publicacion(config, {'accion': 'publicacion',
                                       'version_anterior': anterior['id_ejecucion'] if anterior else None,
                                       'version_vigente': informe['id_ejecucion']})
        except OSError as exc:
            informe['advertencias'].append(f'No se pudo guardar auditoría de publicación: {exc}')
        informe['publicacion'] = 'aprobado_y_publicado'
        mlflow.log_artifact(str(carpeta_final / 'metadatos.json'), artifact_path='version_aprobada')
        logger.info('Versión aprobada y publicada: %s', carpeta_final)
        mlflow.log_param('version_publicada', informe['id_ejecucion'])
        return puntero
    finally:
        if carpeta_temporal.exists():
            shutil.rmtree(carpeta_temporal)


def restaurar(config, identificador):
    """Restaura una versión aprobada sin borrar las versiones posteriores."""
    carpeta = Path(config['rutas']['resultados_aprobados']) / identificador
    datos, metadatos = carpeta / 'estimaciones.csv', carpeta / 'metadatos.json'
    if not datos.is_file() or not metadatos.is_file():
        raise FileNotFoundError('Versión aprobada no encontrada: ' + identificador)
    with metadatos.open(encoding='utf-8') as archivo:
        meta = json.load(archivo)
    if meta.get('id_ejecucion') != identificador or hash_archivo(datos) != meta.get('sha256_estimaciones'):
        raise ValueError('La versión solicitada no supera la comprobación de integridad')
    puntero = {'id_ejecucion': identificador, 'mlflow_run_id': meta['mlflow_run_id'],
               'ruta_estimaciones': str(datos), 'ruta_metadatos': str(metadatos),
               'fecha_actualizacion': ahora(), 'accion': 'restauracion'}
    anterior = leer_version_vigente(config)
    guardar_json(puntero, config['rutas']['version_vigente'])
    try:
        auditar_publicacion(config, {'accion': 'restauracion',
                                   'version_anterior': anterior['id_ejecucion'] if anterior else None,
                                   'version_vigente': identificador})
    except OSError as exc:
        print(f'ADVERTENCIA: no se pudo guardar auditoría de restauración: {exc}', file=sys.stderr)
    return puntero


def comparar(config, identificador_a, identificador_b):
    """Verifica dos ejecuciones con los mismos datos y configuración."""
    reportes = Path(config['rutas']['reportes'])
    with (reportes / identificador_a / 'ejecucion.json').open(encoding='utf-8') as archivo:
        a = json.load(archivo)
    with (reportes / identificador_b / 'ejecucion.json').open(encoding='utf-8') as archivo:
        b = json.load(archivo)
    if a['origen']['sha256'] != b['origen']['sha256'] or a['config_sha256'] != b['config_sha256']:
        raise ValueError('Las ejecuciones no utilizaron los mismos datos y configuración')
    ruta_a = Path(config['rutas']['resultados_candidatos']) / identificador_a / 'estimaciones.csv'
    ruta_b = Path(config['rutas']['resultados_candidatos']) / identificador_b / 'estimaciones.csv'
    valores_a = pd.read_csv(ruta_a).sort_values(['via', 'fecha'])
    valores_b = pd.read_csv(ruta_b).sort_values(['via', 'fecha'])
    if not valores_a[['via', 'fecha']].reset_index(drop=True).equals(valores_b[['via', 'fecha']].reset_index(drop=True)):
        raise ValueError('Las ejecuciones no generaron las mismas claves de salida')
    diferencias = np.abs(valores_a['estimacion'].to_numpy(dtype=float) - valores_b['estimacion'].to_numpy(dtype=float))
    max_diferencia = float(np.max(diferencias)) if len(diferencias) else float('inf')
    tolerancia = float(config['aceptacion']['tolerancia_reproducibilidad'])
    claves_metricas = set(a['metricas']) | set(b['metricas'])
    mismas_metricas = set(a['metricas']) == set(b['metricas'])
    error_metricas = (max(abs(a['metricas'][k] - b['metricas'][k]) for k in claves_metricas)
                      if mismas_metricas and claves_metricas else 0 if mismas_metricas else float('inf'))
    mismos_modelos = ({k: (v['tipo'], v['hiperparametros']) for k, v in a['modelos'].items()}
                     == {k: (v['tipo'], v['hiperparametros']) for k, v in b['modelos'].items()})
    resultado = {'ejecuciones': [identificador_a, identificador_b], 'misma_fuente': True,
                 'misma_configuracion': True, 'mismo_commit': a['git_commit'] == b['git_commit'],
                 'mismas_dependencias': a['dependencias'] == b['dependencias'],
                 'mismos_modelos': mismos_modelos, 'diferencia_maxima': max_diferencia,
                 'diferencia_maxima_metricas': error_metricas, 'tolerancia_absoluta': tolerancia,
                 'equivalentes': bool(mismos_modelos and max_diferencia <= tolerancia
                                      and error_metricas <= tolerancia)}
    guardar_json(resultado, reportes / f'comparacion_{identificador_a}_{identificador_b}.json')
    return resultado


def ejecutar(config, ruta_config):
    mlflow = importar_mlflow()
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config['mlflow']['experimento'])
    identificador = id_ejecucion()
    carpeta, informe = preparar_carpeta_y_reporte(config, identificador)
    informe['config_sha256'] = hash_archivo(ruta_config)
    informe['dependencias'] = informacion_dependencias()
    informe['git_commit'] = git_commit(ruta_config.parent)
    informe['git_modificaciones'] = git_modificaciones(ruta_config.parent)
    informe['huellas_codigo'] = huellas_codigo(Path(__file__).resolve().parent)
    logger = configurar_logging(config['rutas']['logs'], identificador)
    archivo_log = Path(config['rutas']['logs']) / f'pipeline_{identificador}.log'
    codigo_error = None
    try:
        with mlflow.start_run(run_name=identificador) as run:
            informe['mlflow_run_id'] = run.info.run_id
            mlflow.set_tags({'id_ejecucion': identificador,
                             'git_commit': informe['git_commit'] or 'no_disponible',
                             'git_sucio': str(bool(informe['git_modificaciones']))
                                          if informe['git_modificaciones'] is not None else 'desconocido',
                             'estado_pipeline': 'ejecutando', 'destino': config['publicacion'].get('destino') or 'no_configurado'})
            mlflow.log_params({'config_sha256': informe['config_sha256'],
                               'version_procedimiento': config['publicacion']['version_procedimiento'],
                               'estrategia_referencia': config['modelos']['referencia']['estrategia'],
                               'mae_maximo': str(config['aceptacion']['mae_maximo']),
                               'factor_reparto': str(config['publicacion'].get('factor_reparto')),
                               'semilla': config['modelos']['semilla'],
                               'inicio_validacion': str(config['modelos']['validacion']['inicio_validacion']),
                               'fin_validacion': str(config['modelos']['validacion']['fin_validacion']),
                               'inicio_prueba': str(config['modelos']['validacion']['inicio_prueba']),
                               'fin_prueba': str(config['modelos']['validacion']['fin_prueba']),
                               'horizonte': config['modelos']['pronostico']['horizonte_meses']})
            if informe['git_commit'] is None:
                aviso = 'No se encontró commit de Git; para cumplir el laboratorio debes ejecutar desde un repositorio versionado'
                informe['advertencias'].append(aviso)
                logger.warning(aviso)
            elif informe['git_modificaciones']:
                aviso = 'Hay cambios sin confirmar en Git; revisa huellas_codigo del reporte para garantizar trazabilidad'
                informe['advertencias'].append(aviso)
                logger.warning(aviso)
            try:
                datos = etapa(informe, logger, 'lectura', lambda: leer_fuente(config, informe, mlflow))
                limpio = etapa(informe, logger, 'limpieza_y_controles',
                               lambda: limpiar_y_validar(datos, config, informe, mlflow, logger, carpeta))
                pred = etapa(informe, logger, 'modelado_y_evaluacion',
                             lambda: entrenar_evaluar_pronosticar(limpio, config, informe, mlflow, logger, carpeta))
                salida = añadir_metadatos(pred, config, informe)
                carpeta_candidata = Path(config['rutas']['resultados_candidatos']) / identificador
                ruta_candidata = carpeta_candidata / 'estimaciones.csv'
                guardar_datos(salida, ruta_candidata)
                informe['archivos']['estimaciones_candidatas'] = str(ruta_candidata)
                control_salida = validacion_estimaciones(salida, config)
                registrar_control('estimaciones', control_salida, informe, mlflow, logger)
                calidad = etapa(informe, logger, 'monitoreo',
                                lambda: monitorear(pred, limpio, config, informe))
                informe['aceptacion'] = decidir_aceptacion(config, informe, salida, control_salida)
                guardar_json(informe['aceptacion'], carpeta / 'aceptacion.json')
                mlflow.log_metric('version_aceptada', int(informe['aceptacion']['aprobado']))
                if informe['aceptacion']['aprobado']:
                    etapa(informe, logger, 'publicacion',
                          lambda: publicar_version(config, informe, ruta_candidata, calidad, mlflow, logger))
                    informe['estado'] = 'publicado'
                else:
                    informe['estado'] = 'rechazado'
                    informe['errores'].extend(informe['aceptacion']['razones'])
                    logger.warning('Candidata rechazada: %s', informe['aceptacion']['razones'])
                mlflow.set_tag('estado_pipeline', informe['estado'])
                mlflow.set_tag('publicacion', informe['publicacion'])
            except Exception as exc:
                codigo_error = exc
                informe['estado'] = 'fallido'
                informe['errores'].append(f'{type(exc).__name__}: {exc}')
                guardar_json({'error': str(exc), 'tipo': type(exc).__name__,
                              'traceback': traceback.format_exc()}, carpeta / 'error.json')
                logger.exception('Ejecución interrumpida')
                mlflow.set_tag('estado_pipeline', 'fallido')
                mlflow.set_tag('publicacion', 'no_publicado')
            finally:
                informe['fin'] = ahora()
                guardar_json(informe, carpeta / 'ejecucion.json')
                shutil.copy2(ruta_config, carpeta / 'config_ejecutada.yaml')
                guardar_json(informe['dependencias'], carpeta / 'dependencias.json')
                for archivo in carpeta.rglob('*'):
                    if archivo.is_file():
                        mlflow.log_artifact(str(archivo), artifact_path='evidencia')
                if archivo_log.is_file():
                    for handler in logger.handlers:
                        handler.flush()
                    mlflow.log_artifact(str(archivo_log), artifact_path='logs')
                ruta_candidata = Path(config['rutas']['resultados_candidatos']) / identificador / 'estimaciones.csv'
                if ruta_candidata.is_file():
                    mlflow.log_artifact(str(ruta_candidata), artifact_path='candidatos')
            if codigo_error is not None:
                # Marcar como FAILED en MLflow sin perder los artifacts ya registrados.
                raise codigo_error
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    return informe


def main():
    parser = argparse.ArgumentParser(description='Pipeline reproducible del Observatorio de Turismo')
    parser.add_argument('--config', default='config.yaml', help='Ruta del YAML de configuración')
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument('--restaurar', metavar='ID', help='Restaurar una versión aprobada')
    grupo.add_argument('--comparar', nargs=2, metavar=('ID_1', 'ID_2'),
                       help='Comparar reproducibilidad de dos ejecuciones')
    args = parser.parse_args()
    config, ruta_config = leer_config(args.config)
    if args.restaurar:
        print(json.dumps(restaurar(config, args.restaurar), indent=2, ensure_ascii=False))
    elif args.comparar:
        print(json.dumps(comparar(config, *args.comparar), indent=2, ensure_ascii=False))
    else:
        informe = ejecutar(config, ruta_config)
        print(json.dumps({'id_ejecucion': informe['id_ejecucion'],
                          'mlflow_run_id': informe['mlflow_run_id'],
                          'estado': informe['estado'],
                          'publicacion': informe['publicacion']}, indent=2, ensure_ascii=False))
        return 0 if informe['estado'] == 'publicado' else 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
