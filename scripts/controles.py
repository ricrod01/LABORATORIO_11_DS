from pathlib import Path

import numpy as np
import pandas as pd


def _resultado(df):
    return {'aprobado': True, 'errores': [], 'advertencias': [], 'registros': len(df), 'detalle': {}}


def _finalizar(resultado):
    resultado['aprobado'] = len(resultado['errores']) == 0
    return resultado


def _normalizar(serie):
    return (serie.astype('string').str.strip().str.normalize('NFKD')
            .str.replace(r'[\u0300-\u036f]', '', regex=True).str.upper())


# Valida la fuente original ANTES de modificar filas o convertir tipos.
def validacion_inicial(df, config):
    reglas = config['controles']['inicial']
    nombres = config['limpieza']['columnas']
    resultado = _resultado(df)
    faltan = sorted(set(nombres) - set(df.columns))
    if faltan:
        resultado['errores'].append(f'Columnas originales faltantes: {faltan}')
    if df.empty:
        resultado['errores'].append('La fuente original está vacía')
    if resultado['errores']:
        return _finalizar(resultado)

    # Duplicados completamente idénticos en la fuente se reportan aparte.
    # No se borran automáticamente, porque podrían modificar el total.
    duplicados_exactos = int(df.duplicated().sum())
    resultado['detalle']['duplicados_exactos_fuente'] = duplicados_exactos
    if duplicados_exactos:
        severidad = reglas.get('duplicados_exactos_severidad', 'error')
        if severidad not in ('error', 'advertencia'):
            raise ValueError('duplicados_exactos_severidad debe ser error o advertencia')
        resultado['errores' if severidad == 'error' else 'advertencias'].append(
            f'Fuente original: {duplicados_exactos} filas completamente idénticas; comprobar si son repeticiones válidas'
        )
    permitidos = set(reglas.get('columnas_faltantes_permitidas', []))
    for columna in nombres:
        nulos = int(df[columna].isna().sum())
        resultado['detalle'][f'nulos_{columna}'] = nulos
        if nulos:
            campo = 'advertencias' if columna in permitidos else 'errores'
            resultado[campo].append(f'{columna}: {nulos} valores faltantes')

    for original in ['Vía', 'País', 'Tipo de Viajero']:
        columna = next((c for c, nuevo in nombres.items()
                        if nuevo == {'Vía': 'via', 'País': 'pais', 'Tipo de Viajero': 'tipo_viajero'}[original]), None)
        if columna is not None:
            incompatibles = int((~df[columna].dropna().map(lambda x: isinstance(x, str))).sum())
            if incompatibles:
                resultado['errores'].append(f'{columna}: {incompatibles} valores no textuales')

    originales = {nuevo: original for original, nuevo in nombres.items()}
    anios = pd.to_numeric(df[originales['anio']], errors='coerce')
    meses = pd.to_numeric(df[originales['mes']], errors='coerce')
    fechas = pd.to_datetime(dict(year=anios, month=meses, day=1), errors='coerce')
    invalidas = (anios.isna() | meses.isna() | ~np.isfinite(anios) | ~np.isfinite(meses)
                 | (anios < reglas['anio_minimo']) | (anios % 1 != 0)
                 | ~meses.between(reglas['mes_minimo'], reglas['mes_maximo'])
                 | (meses % 1 != 0) | fechas.isna())
    resultado['detalle']['fechas_invalidas'] = int(invalidas.sum())
    if invalidas.any():
        resultado['errores'].append(f'Fechas inválidas: {int(invalidas.sum())} registros')

    cantidad = pd.to_numeric(df[originales['cantidad_viajero']], errors='coerce')
    invalida = (cantidad.isna() | ~np.isfinite(cantidad)
                | (cantidad < reglas['cantidad_minima']))
    decimales = int((cantidad.notna() & np.isfinite(cantidad) & (cantidad % 1 != 0)).sum())
    resultado['detalle']['cantidades_invalidas'] = int(invalida.sum())
    resultado['detalle']['cantidades_decimales'] = decimales
    if invalida.any():
        resultado['errores'].append(f'Cantidades inválidas: {int(invalida.sum())} registros')
    if decimales:
        resultado['advertencias'].append(f'Se conservan {decimales} cantidades decimales de la fuente')
    return _finalizar(resultado)


# Valida estructura, integridad, unicidad y cobertura de los datos limpios.
def validacion_final(df, config):
    reglas = config['controles']['final']
    resultado = _resultado(df)
    obligatorias = reglas['columnas_obligatorias']
    faltan = sorted(set(obligatorias) - set(df.columns))
    if faltan:
        resultado['errores'].append(f'Columnas procesadas faltantes: {faltan}')
    if df.empty:
        resultado['errores'].append('Los datos procesados están vacíos')
    if resultado['errores']:
        return _finalizar(resultado)

    if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        resultado['errores'].append('fecha debe tener tipo datetime')
    if not pd.api.types.is_numeric_dtype(df['cantidad_viajero']):
        resultado['errores'].append('cantidad_viajero debe tener tipo numérico')
    for columna in reglas['columnas_texto']:
        if columna not in df:
            resultado['errores'].append(f'Falta columna textual: {columna}')
        elif not df[columna].dropna().map(lambda valor: isinstance(valor, str)).all():
            resultado['errores'].append(f'{columna} contiene valores no textuales')
    if resultado['errores']:
        return _finalizar(resultado)

    permitidos = set(reglas.get('columnas_faltantes_permitidas', []))
    for columna in obligatorias:
        nulos = int(df[columna].isna().sum())
        resultado['detalle'][f'nulos_{columna}'] = nulos
        if nulos:
            campo = 'advertencias' if columna in permitidos else 'errores'
            resultado[campo].append(f'{columna}: {nulos} valores faltantes')

    vias_configuradas = set(config['modelos']['vias'])
    vias_encontradas = set(df['via'].dropna().unique().tolist())
    if vias_encontradas != vias_configuradas:
        resultado['errores'].append(
            f'Vías limpias {sorted(vias_encontradas)} no coinciden con las configuradas {sorted(vias_configuradas)}'
        )

    fechas = df['fecha'].dropna()
    fecha_invalida = fechas.ne(fechas.dt.to_period('M').dt.to_timestamp())
    if fecha_invalida.any():
        resultado['errores'].append(f'Fechas distintas del primer día del mes: {int(fecha_invalida.sum())}')

    clave = reglas['clave_unicidad']
    if not set(clave).issubset(df.columns):
        resultado['errores'].append('La clave de unicidad incluye columnas inexistentes')
    else:
        duplicados = int(df.duplicated(subset=clave).sum())
        resultado['detalle']['duplicados'] = duplicados
        if duplicados:
            resultado['errores'].append(f'Claves duplicadas: {duplicados}')

    cantidades = pd.to_numeric(df['cantidad_viajero'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    invalidas = int((~np.isfinite(cantidades) | (cantidades < reglas['cantidad_minima'])).sum())
    if invalidas:
        resultado['errores'].append(f'Cantidades negativas o no finitas: {invalidas}')

    tipos_permitidos = set(_normalizar(pd.Series(config['limpieza']['tipos_viajero'], dtype='string')))
    tipos_invalidos = int((~_normalizar(df['tipo_viajero']).isin(tipos_permitidos) & df['tipo_viajero'].notna()).sum())
    if tipos_invalidos:
        resultado['errores'].append(f'Tipos de viajero inválidos: {tipos_invalidos}')
    paises_excluidos = set(_normalizar(pd.Series(config['limpieza']['paises_excluidos'], dtype='string')))
    excluidos = int(_normalizar(df['pais']).isin(paises_excluidos).sum())
    if excluidos:
        resultado['errores'].append(f'Países excluidos presentes: {excluidos}')

    cobertura = reglas['cobertura']
    inicio = cobertura.get('fecha_inicio')
    fin = cobertura.get('fecha_fin')
    if inicio is None or fin is None:
        resultado['advertencias'].append('Cobertura sin límites explícitos: puede omitir meses extremos')
    if not fechas.empty:
        desde = pd.Period(inicio, freq='M') if inicio else fechas.min().to_period('M')
        hasta = pd.Period(fin, freq='M') if fin else fechas.max().to_period('M')
        if desde > hasta:
            resultado['errores'].append('El inicio de cobertura es posterior al fin')
        else:
            esperados = pd.period_range(desde, hasta, freq='M')
            excepciones_por_via = cobertura.get('periodos_sin_cobertura_permitidos_por_via', {})
            vias = cobertura.get('vias_esperadas')
            if not vias:
                resultado['advertencias'].append('Las vías esperadas se infieren del archivo')
                vias = df['via'].dropna().unique().tolist()
            severidad = cobertura['severidad_faltantes']
            if severidad not in ('error', 'advertencia'):
                raise ValueError('severidad_faltantes debe ser error o advertencia')
            campo = 'errores' if severidad == 'error' else 'advertencias'
            for via in vias:
                excepciones = set(excepciones_por_via.get(via, []))
                meses = pd.PeriodIndex(df.loc[df['via'] == via, 'fecha'].dropna().dt.to_period('M').unique(), freq='M')
                ausentes = esperados.difference(meses)
                excusados = [str(mes) for mes in ausentes if str(mes) in excepciones]
                criticos = [str(mes) for mes in ausentes if str(mes) not in excepciones]
                resultado['detalle'][f'cobertura_{via}'] = {
                    'esperados': len(esperados), 'faltantes_criticos': len(criticos),
                    'faltantes_permitidos': len(excusados)
                }
                if criticos:
                    resultado[campo].append(f'{via}: faltan {len(criticos)} meses: {criticos}')
                if excusados:
                    resultado['advertencias'].append(f'{via}: {len(excusados)} meses sin cobertura declarados: {excusados}')
    else:
        resultado['errores'].append('No hay fechas válidas para comprobar cobertura')
    return _finalizar(resultado)


# Verifica que se utilizó la fuente mensual autorizada y la granularidad requerida.
def validacion_metodologica(df, config, metadatos_fuente):
    resultado = _resultado(df)
    reglas = config['controles']['metodologia']
    ruta_permitida = Path(config['rutas']['datos_originales']).resolve()
    ruta_usada = metadatos_fuente.get('ruta') if metadatos_fuente else None
    frecuencia = metadatos_fuente.get('frecuencia') if metadatos_fuente else None
    if not ruta_usada or Path(ruta_usada).resolve() != ruta_permitida:
        resultado['errores'].append('No se acredita la lectura de la fuente mensual autorizada')
    if frecuencia != reglas['frecuencia_esperada']:
        resultado['errores'].append('La frecuencia declarada de la fuente no es mensual')
    if 'fecha' not in df or not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        resultado['errores'].append('No existe fecha mensual válida')
    elif df['fecha'].isna().any() or not df['fecha'].eq(df['fecha'].dt.to_period('M').dt.to_timestamp()).all():
        resultado['errores'].append('La fecha no representa el inicio de cada mes')
    if 'imputado_pandemia' in df or 'valor_observado' in df:
        resultado['errores'].append('Los datos limpios no deben sustituir observaciones por imputaciones')
    resultado['detalle']['fuente_utilizada'] = str(ruta_usada)
    resultado['detalle']['frecuencia_declarada'] = frecuencia
    return _finalizar(resultado)


# Control crítico sobre los resultados candidatos, antes de publicarlos.
def validacion_estimaciones(df, config):
    resultado = _resultado(df)
    requeridas = ['via', 'fecha', 'estimacion']
    faltan = sorted(set(requeridas) - set(df.columns))
    if faltan or df.empty:
        resultado['errores'].append(f'Estimaciones vacías o columnas ausentes: {faltan}')
        return _finalizar(resultado)
    if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
        resultado['errores'].append('fecha de estimaciones debe ser datetime')
        return _finalizar(resultado)
    if df[requeridas].isna().any().any():
        resultado['errores'].append('Hay campos obligatorios vacíos en las estimaciones')
    duplicados = int(df.duplicated(['via', 'fecha']).sum())
    if duplicados:
        resultado['errores'].append(f'Estimaciones duplicadas por vía y mes: {duplicados}')
    valores = pd.to_numeric(df['estimacion'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    invalidos = int((~np.isfinite(valores) | (valores < 0)).sum())
    if invalidos:
        resultado['errores'].append(f'Estimaciones negativas o no finitas: {invalidos}')
    if not df['fecha'].dropna().eq(df['fecha'].dropna().dt.to_period('M').dt.to_timestamp()).all():
        resultado['errores'].append('Existen fechas distintas del primer día del mes')
    pronostico_config = config['modelos']['pronostico']
    meses_esperados = pronostico_config['horizonte_meses']
    inicio_configurado = pronostico_config.get('fecha_inicio')
    periodo_inicio = pd.Period(inicio_configurado, freq='M') if inicio_configurado else None
    vias_esperadas = set(config['modelos']['vias'])
    presentes = set(df['via'].dropna())
    if presentes != vias_esperadas:
        resultado['errores'].append(f'Vías distintas a las esperadas: {sorted(presentes)}')
    fechas_referencia = None
    for via in config['modelos']['vias']:
        meses = pd.PeriodIndex(df.loc[df['via'] == via, 'fecha'].dropna().dt.to_period('M').unique(), freq='M').sort_values()
        if len(meses) != meses_esperados or (len(meses) and not meses.equals(pd.period_range(meses.min(), periods=len(meses), freq='M'))):
            resultado['errores'].append(f'{via}: meses incompletos o no consecutivos')
        if periodo_inicio is not None and len(meses) and meses.min() != periodo_inicio:
            resultado['errores'].append(f'{via}: el primer mes no coincide con la fecha de pronóstico configurada')
        if fechas_referencia is None:
            fechas_referencia = meses
        elif not meses.equals(fechas_referencia):
            resultado['errores'].append(f'{via}: los meses no coinciden con las otras vías')
    return _finalizar(resultado)
