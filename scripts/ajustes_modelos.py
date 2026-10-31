from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Obtiene una serie mensual, sin reemplazar meses ausentes por cero.
def agrupa_via(df, via, config):
    parametros = config['modelos']
    columna_via = parametros['columna_via']
    columna_fecha = parametros['columna_fecha']
    variable = parametros['variable_objetivo']
    if via not in parametros['vias']:
        raise ValueError(f'Vía no configurada: {via}')
    datos = df.loc[df[columna_via] == via, [columna_fecha, variable]].copy()
    if datos.empty:
        raise ValueError(f'No hay datos para la vía {via}')
    if datos.isna().any().any():
        raise ValueError(f'Hay fechas o cantidades faltantes para {via}')
    serie = datos.groupby(columna_fecha)[variable].sum(min_count=1).sort_index().to_frame()
    serie.index = pd.DatetimeIndex(serie.index).to_period('M')
    esperados = pd.period_range(serie.index.min(), serie.index.max(), freq='M')
    faltantes = esperados.difference(serie.index)
    pandemia = parametros['pandemia']
    if pandemia['aplicar']:
        inicio = pd.Period(pandemia['inicio'], freq='M')
        fin = pd.Period(pandemia['fin'], freq='M')
        faltantes = faltantes[(faltantes < inicio) | (faltantes > fin)]
    if len(faltantes):
        raise ValueError(f'Hay meses sin cobertura fuera del ajuste autorizado en {via}: {list(map(str, faltantes))}')
    return serie


# Ajusta el período excepcional y preserva por separado los valores observados.
def ajuste_pandemia(serie, config):
    reglas = config['modelos']['pandemia']
    variable = config['modelos']['variable_objetivo']
    resultado = serie.copy().sort_index()
    if resultado.empty or not isinstance(resultado.index, pd.PeriodIndex):
        raise ValueError('Se requiere una serie mensual con PeriodIndex')
    resultado['valor_observado'] = resultado[variable].copy()
    resultado['imputado_pandemia'] = False
    if not reglas['aplicar']:
        return resultado

    inicio = pd.Period(reglas['inicio'], freq='M')
    fin = pd.Period(reglas['fin'], freq='M')
    if inicio > fin:
        raise ValueError('Las fechas del ajuste de pandemia están invertidas')
    if resultado.index.max() < inicio or resultado.index.min() > fin:
        return resultado

    anteriores = resultado.loc[resultado.index < inicio, variable]
    if len(anteriores) < reglas['minimo_meses'] or anteriores.index.max() != inicio - 1:
        raise ValueError('No hay historial continuo suficiente antes de la pandemia')
    esperado = pd.period_range(anteriores.index.min(), inicio - 1, freq='M')
    if not anteriores.index.equals(esperado) or anteriores.isna().any():
        raise ValueError('El historial anterior a la pandemia está incompleto')

    try:
        from sktime.forecasting.naive import NaiveForecaster
    except ImportError as exc:
        raise ImportError('Para el ajuste de pandemia debes instalar sktime') from exc

    meses = pd.period_range(inicio, fin, freq='M')
    modelo = NaiveForecaster(strategy=reglas['estrategia'], sp=reglas['periodicidad'])
    modelo.fit(anteriores.astype(float))
    predicciones = modelo.predict(fh=list(range(1, len(meses) + 1)))
    if not predicciones.index.equals(meses) or not np.isfinite(predicciones.to_numpy(dtype=float)).all():
        raise ValueError('El pronóstico de pandemia no coincide con las fechas o no es finito')
    resultado = resultado.reindex(resultado.index.union(meses)).sort_index()
    resultado.loc[meses, variable] = predicciones.to_numpy(dtype=float)
    resultado['imputado_pandemia'] = resultado.index.isin(meses)
    if resultado[variable].isna().any():
        raise ValueError('Existen meses sin cobertura fuera del período de ajuste')
    return resultado


# Define entrenamiento, validación de modelos y prueba final sin mezclar períodos.
def separacion_temporal(serie, config):
    reglas = config['modelos']['validacion']
    campos = ['inicio_validacion', 'fin_validacion', 'inicio_prueba', 'fin_prueba']
    if any(reglas.get(campo) is None for campo in campos):
        raise ValueError('Debes definir las cuatro fechas de validación y prueba en config.yaml')
    inicio_val, fin_val, inicio_test, fin_test = (
        pd.Period(reglas[campo], freq='M') for campo in campos
    )
    if not inicio_val <= fin_val < inicio_test <= fin_test or fin_val + 1 != inicio_test:
        raise ValueError('Los períodos de validación y prueba deben ser ordenados y consecutivos')
    entrenamiento = serie.loc[serie.index < inicio_val].copy()
    validacion = serie.loc[inicio_val:fin_val].copy()
    prueba = serie.loc[inicio_test:fin_test].copy()
    partes = [('entrenamiento', entrenamiento), ('validación', validacion), ('prueba', prueba)]
    for nombre, datos in partes:
        if datos.empty or not datos.index.equals(pd.period_range(datos.index.min(), datos.index.max(), freq='M')):
            raise ValueError(f'Período de {nombre} vacío o con meses faltantes')
    if (entrenamiento.index.max() + 1 != inicio_val or validacion.index.min() != inicio_val
            or validacion.index.max() != fin_val or prueba.index.min() != inicio_test
            or prueba.index.max() != fin_test):
        raise ValueError('Los períodos configurados no están completamente cubiertos')
    for nombre, datos in [('validación', validacion), ('prueba', prueba)]:
        if 'imputado_pandemia' in datos and datos['imputado_pandemia'].fillna(False).any():
            raise ValueError(f'No se pueden evaluar valores imputados en {nombre}')
        if 'valor_observado' in datos and datos['valor_observado'].isna().any():
            raise ValueError(f'Faltan observaciones reales en {nombre}')
    return entrenamiento, validacion, prueba


def _ajustar_tipo(y_train, tipo_modelo, hiperparametros, config):
    if tipo_modelo == 'SARIMA':
        p = hiperparametros
        modelo = SARIMAX(
            y_train, order=(p['p'], p['d'], p['q']),
            seasonal_order=(p['P'], p['D'], p['Q'], p['S']),
            trend=p['trend'], enforce_stationarity=p['enforce_stationarity'],
            enforce_invertibility=p['enforce_invertibility']
        ).fit(maxiter=p['maxiter'], disp=False)
        if not modelo.mle_retvals.get('converged', True):
            raise RuntimeError('SARIMA no convergió')
        return modelo
    if tipo_modelo == 'Prophet':
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError('Debes instalar prophet para utilizar este modelo') from exc
        np.random.seed(config['modelos']['semilla'])
        datos = pd.DataFrame({'ds': y_train.index.to_timestamp(), 'y': y_train.to_numpy()})
        return Prophet(**hiperparametros).fit(datos)
    if tipo_modelo == 'Holt-Winters':
        return ExponentialSmoothing(
            y_train, trend=hiperparametros['trend'], seasonal=hiperparametros['seasonal'],
            seasonal_periods=hiperparametros['seasonal_periods']
        ).fit()
    if tipo_modelo == 'Suavizamiento exponencial simple':
        return SimpleExpSmoothing(y_train).fit(optimized=hiperparametros['optimized'])
    raise ValueError(f'Modelo no reconocido: {tipo_modelo}')


def _pronosticar(modelo, tipo_modelo, meses):
    if tipo_modelo == 'Prophet':
        futuro = pd.DataFrame({'ds': meses.to_timestamp()})
        valores = modelo.predict(futuro)['yhat'].to_numpy(dtype=float)
    else:
        valores = np.asarray(modelo.forecast(steps=len(meses)), dtype=float)
    if len(valores) != len(meses) or not np.isfinite(valores).all():
        raise ValueError(f'Predicciones inválidas del modelo {tipo_modelo}')
    return pd.Series(valores, index=meses, name='estimacion')


def _verificar_periodos(serie_entrenamiento, serie_evaluacion, variable):
    if serie_entrenamiento.empty or serie_evaluacion.empty:
        raise ValueError('Los conjuntos de entrenamiento y evaluación no pueden estar vacíos')
    if (serie_entrenamiento[variable].isna().any() or serie_evaluacion[variable].isna().any()
            or not np.isfinite(serie_entrenamiento[variable].to_numpy(dtype=float)).all()
            or not np.isfinite(serie_evaluacion[variable].to_numpy(dtype=float)).all()):
        raise ValueError('Existen valores faltantes o no finitos')
    for datos in [serie_entrenamiento, serie_evaluacion]:
        if not datos.index.equals(pd.period_range(datos.index.min(), datos.index.max(), freq='M')):
            raise ValueError('Hay meses faltantes dentro de entrenamiento o evaluación')
    if serie_entrenamiento.index.max() + 1 != serie_evaluacion.index.min():
        raise ValueError('Entrenamiento y evaluación deben ser consecutivos')
    if 'imputado_pandemia' in serie_evaluacion and serie_evaluacion['imputado_pandemia'].fillna(False).any():
        raise ValueError('No se permite evaluar contra valores imputados')
    if 'valor_observado' in serie_evaluacion and serie_evaluacion['valor_observado'].isna().any():
        raise ValueError('La evaluación requiere valores observados')


# Selecciona hiperparámetros SOLO sobre el conjunto de validación.
def entrenamiento_modelo(serie_entrenamiento, serie_validacion, config):
    parametros = config['modelos']
    variable = parametros['variable_objetivo']
    rejilla = parametros['sarima']
    _verificar_periodos(serie_entrenamiento, serie_validacion, variable)
    np.random.seed(parametros['semilla'])
    y_train = serie_entrenamiento[variable].astype(float)
    y_val = serie_validacion[variable].astype(float)
    seleccion = {'modelo': None, 'tipo': None, 'mae': np.inf, 'hiperparametros': {}}
    fallos = []

    def intentar(tipo, hiperparametros):
        try:
            modelo = _ajustar_tipo(y_train, tipo, hiperparametros, config)
            pronostico = _pronosticar(modelo, tipo, y_val.index)
            mae = float(mean_absolute_error(y_val, pronostico))
            if mae < seleccion['mae']:
                seleccion.update(modelo=modelo, tipo=tipo, mae=mae,
                                 hiperparametros=hiperparametros.copy())
        except Exception as exc:
            fallos.append(f'{tipo} {hiperparametros}: {type(exc).__name__}: {exc}')

    habilitados = parametros['habilitados']
    if 'SARIMA' in habilitados:
        for p, d, q, P, D, Q in product(rejilla['p'], rejilla['d'], rejilla['q'],
                                        rejilla['P'], rejilla['D'], rejilla['Q']):
            intentar('SARIMA', {
                'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 'S': rejilla['S'],
                'trend': rejilla['trend'], 'enforce_stationarity': rejilla['enforce_stationarity'],
                'enforce_invertibility': rejilla['enforce_invertibility'], 'maxiter': rejilla['maxiter']
            })
    for tipo, clave in [('Prophet', 'prophet'), ('Holt-Winters', 'holt_winters'),
                        ('Suavizamiento exponencial simple', 'suavizamiento_simple')]:
        if tipo in habilitados:
            intentar(tipo, parametros[clave])
    if seleccion['modelo'] is None:
        raise RuntimeError(f'Ningún modelo pudo ajustarse. Fallos: {fallos}')
    return (seleccion['modelo'], seleccion['tipo'], seleccion['mae'],
            seleccion['hiperparametros'], fallos)


# Reajusta el modelo elegido con los períodos permitidos antes del mes evaluado.
def ajustar_modelo(serie_entrenamiento, tipo_modelo, hiperparametros, config):
    variable = config['modelos']['variable_objetivo']
    if serie_entrenamiento.empty or serie_entrenamiento[variable].isna().any():
        raise ValueError('No existen datos completos para reajustar el modelo')
    return _ajustar_tipo(serie_entrenamiento[variable].astype(float),
                        tipo_modelo, hiperparametros, config)


# Referencia basada exclusivamente en observaciones anteriores.
def modelo_referencia(serie_entrenamiento, meses, config):
    variable = config['modelos']['variable_objetivo']
    reglas = config['modelos']['referencia']
    observados = (serie_entrenamiento['valor_observado'] if 'valor_observado' in serie_entrenamiento
                  else serie_entrenamiento[variable])
    observados = observados.dropna().astype(float)
    observados = observados.loc[observados.index < meses.min()]
    if observados.empty:
        raise ValueError('No hay observaciones anteriores para el modelo de referencia')
    if reglas['estrategia'] == 'ultimo':
        valor = float(observados.iloc[-1])
    elif reglas['estrategia'] == 'promedio_historico':
        ventana = reglas['ventana_meses']
        valor = float(observados.tail(ventana).mean())
    else:
        raise ValueError('Estrategia de referencia no reconocida')
    return pd.Series(valor, index=meses, name='referencia')


# Compara modelo seleccionado y referencia sobre los mismos meses observados.
def evaluar_modelo(modelo, tipo_modelo, serie_anterior, serie_prueba, config):
    variable = config['modelos']['variable_objetivo']
    _verificar_periodos(serie_anterior, serie_prueba, variable)
    meses = serie_prueba.index
    pronostico = _pronosticar(modelo, tipo_modelo, meses)
    referencia = modelo_referencia(serie_anterior, meses, config)
    observados = (serie_prueba['valor_observado'] if 'valor_observado' in serie_prueba
                  else serie_prueba[variable]).astype(float)
    return {
        'mae_modelo': float(mean_absolute_error(observados, pronostico)),
        'mae_referencia': float(mean_absolute_error(observados, referencia)),
        'meses_evaluados': [str(mes) for mes in meses],
        'comparacion': pd.DataFrame({'observado': observados, 'estimacion': pronostico,
                                    'referencia': referencia})
    }


# Pronostica los próximos meses tras reajustar con todo el historial permitido.
def generacion_estimaciones(modelo, tipo_modelo, ultimo_mes, via, config):
    horizonte = config['modelos']['pronostico']['horizonte_meses']
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError('El horizonte de pronóstico debe ser un entero positivo')
    if via not in config['modelos']['vias']:
        raise ValueError(f'Vía no permitida: {via}')
    ultimo_mes = pd.Period(ultimo_mes, freq='M')
    meses = pd.period_range(ultimo_mes + 1, periods=horizonte, freq='M')
    pronostico = _pronosticar(modelo, tipo_modelo, meses)
    return pd.DataFrame({'via': via, 'fecha': meses.to_timestamp(),
                         'estimacion': pronostico.to_numpy()})
