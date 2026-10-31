import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def _normalizar_texto(serie):
    return (serie.astype('string').str.strip().str.normalize('NFKD')
            .str.replace(r'[\u0300-\u036f]', '', regex=True).str.upper())


# Selecciona y renombra columnas sin alterar el DataFrame original.
def columnas_necesarias(df, columnas):
    return df.loc[:, list(columnas)].rename(columns=columnas).copy()


# Filtra después de normalizar, usando las categorías originales del config.
def filtrado_filas(df, tipos_viajero, paises_excluidos, vias_permitidas=None):
    tipos = _normalizar_texto(pd.Series(tipos_viajero, dtype='string')).tolist()
    paises = _normalizar_texto(pd.Series(paises_excluidos, dtype='string')).tolist()
    mascara = (_normalizar_texto(df['tipo_viajero']).isin(tipos)
               & ~_normalizar_texto(df['pais']).isin(paises))
    if vias_permitidas is not None:
        vias = _normalizar_texto(pd.Series(vias_permitidas, dtype='string')).tolist()
        mascara = mascara & _normalizar_texto(df['via']).isin(vias)
    return df.loc[mascara.fillna(False)].copy()


# Genera fechas mensuales (primer día de cada mes).
def columna_fecha(df):
    resultado = df.copy()
    resultado['fecha'] = pd.to_datetime(
        dict(year=resultado['anio'], month=resultado['mes'], day=1), errors='raise'
    )
    return resultado.drop(columns=['anio', 'mes'])


# Conserva la precisión de las cantidades y estandariza las categorías.
def transformacion(df):
    resultado = df.copy()
    cantidad = pd.to_numeric(resultado['cantidad_viajero'], errors='raise')
    if not np.isfinite(cantidad.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError('La cantidad de viajeros contiene nulos o valores no finitos')
    resultado['cantidad_viajero'] = cantidad.astype('float64')
    for columna in ['via', 'frontera', 'pais']:
        resultado[columna] = _normalizar_texto(resultado[columna])
    resultado['tipo_viajero'] = resultado['tipo_viajero'].astype('string').str.strip().str.title()
    return resultado


# Consolida filas que pertenecen a la misma combinación mensual de frontera/país/tipo.
# No elimina duplicados por su posición: suma sus cantidades y registra el detalle.
def consolidar_granularidad(df, clave):
    clave = list(clave)
    requeridas = set(clave) | {'cantidad_viajero'}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f'No se puede consolidar: faltan columnas {sorted(faltantes)}')
    extras = set(df.columns).difference(requeridas)
    if extras:
        raise ValueError(f'Columnas sin regla de consolidación: {sorted(extras)}')
    if df.empty:
        raise ValueError('No hay registros para las vías configuradas')
    if df[clave].isna().any().any():
        raise ValueError('La clave de consolidación tiene valores faltantes')
    original = float(df['cantidad_viajero'].sum())
    grupos = df.groupby(clave, dropna=False, sort=True)
    multiplicidades = grupos.size()
    detalle = {
        'filas_antes': int(len(df)),
        'grupos_repetidos': int((multiplicidades > 1).sum()),
        'filas_adicionales_consolidadas': int((multiplicidades - 1).clip(lower=0).sum()),
        'clave_consolidacion': clave,
        'suma_antes': original
    }
    resultado = grupos['cantidad_viajero'].sum(min_count=1).reset_index()
    detalle['filas_despues'] = int(len(resultado))
    detalle['suma_despues'] = float(resultado['cantidad_viajero'].sum())
    if not np.isclose(detalle['suma_antes'], detalle['suma_despues'], rtol=1e-12, atol=1e-6):
        raise ValueError('La consolidación alteró la cantidad total de viajeros')
    return resultado, detalle


# Utiliza un identificador compartido con el pipeline general.
def configurar_logging(carpeta_logs, identificador=None):
    carpeta = Path(carpeta_logs)
    carpeta.mkdir(parents=True, exist_ok=True)
    identificador = identificador or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logger = logging.getLogger(f'pipeline_{identificador}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formato = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                               datefmt='%Y-%m-%d %H:%M:%S')
    archivo = logging.FileHandler(carpeta / f'pipeline_{identificador}.log', encoding='utf-8')
    consola = logging.StreamHandler()
    for handler in [archivo, consola]:
        handler.setFormatter(formato)
        logger.addHandler(handler)
    logger.info('Inicio de ejecución: %s', identificador)
    return logger


# Escritura atómica para no dejar un CSV parcialmente guardado.
def guardar_datos(df, ruta_archivo):
    ruta = Path(ruta_archivo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', prefix=f'{ruta.stem}_',
                                         dir=ruta.parent, encoding='utf-8', newline='',
                                         delete=False) as archivo:
            temporal = Path(archivo.name)
            df.to_csv(archivo, index=False, sep=',', encoding='utf-8', date_format='%Y-%m-%d')
        os.replace(temporal, ruta)
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()
    return str(ruta)
