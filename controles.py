"""Controles de calidad previos al ajuste y publicacion de resultados."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


POLITICAS = {
    "Columnas y tipos": {
        "verifica": "Existencia de variables indispensables y compatibilidad de tipos.",
        "advertencia_si": "No aplica: una incompatibilidad impide procesar los datos.",
        "detiene_si": "Falta una columna requerida o existe un valor de tipo incompatible.",
    },
    "Fechas y cobertura": {
        "verifica": "Fechas validas y presencia de cada mes requerido por dimension.",
        "advertencia_si": "Hay registros fuera del periodo requerido.",
        "detiene_si": "Hay fechas invalidas o falta un periodo necesario.",
    },
    "Unicidad": {
        "verifica": "Una fila por la clave y granularidad definidas.",
        "advertencia_si": "No aplica: un duplicado puede duplicar el conteo de viajeros.",
        "detiene_si": "Dos o mas filas comparten la clave configurada.",
    },
    "Valores faltantes": {
        "verifica": "Ausencias en variables requeridas y opcionales.",
        "advertencia_si": "La ausencia pertenece a una columna declarada como permitida.",
        "detiene_si": "Falta un valor necesario para estimar.",
    },
    "Valores y denominadores": {
        "verifica": "Conteos no negativos y finitos; denominadores finitos y mayores que cero.",
        "advertencia_si": "No aplica: estos valores producen resultados invalidos.",
        "detiene_si": "Hay negativos, no finitos o un denominador menor o igual que cero.",
    },
    "Regla metodologica": {
        "verifica": "La serie usa observaciones mensuales y, si se configura, la fuente autorizada.",
        "advertencia_si": "No aplica: mezclar granularidades o fuentes cambia el procedimiento.",
        "detiene_si": "Una fecha no representa el inicio de mes o la fuente no es la autorizada.",
    },
}


class ErrorCalidadDatos(RuntimeError):
    """Indica que un control critico impide continuar el pipeline."""


def _columnas_referenciadas(config):
    columnas = set(config.get("tipos", {}))
    for nombre in ("fecha", "columna_fuente"):
        if config.get(nombre):
            columnas.add(config[nombre])
    for nombre in (
        "dimensiones_cobertura",
        "clave",
        "faltantes_permitidos",
        "conteos",
        "denominadores",
    ):
        columnas.update(config.get(nombre, []))
    return columnas


def validar_datos(df, config, ruta_reporte=None):
    """Ejecuta los seis controles de los incisos 3.1 y 3.2.

    Los errores son controles criticos y hacen que ``aprobado`` sea falso. Las
    ausencias expresamente permitidas y los registros fuera del periodo generan
    una advertencia. La funcion nunca completa periodos ausentes con cero.

    Parametros de configuracion requeridos: ``tipos``, ``fecha``, ``inicio``,
    ``fin`` y ``clave``. Son opcionales ``dimensiones_cobertura``,
    ``valores_esperados``, ``faltantes_permitidos``, ``conteos``,
    ``denominadores``, ``columna_fuente`` y ``fuente_autorizada``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")
    if not isinstance(config, dict):
        raise TypeError("config debe ser un diccionario")

    resultados = []

    def registrar(grupo, control, estado, detalle, cantidad=0):
        politica = POLITICAS[grupo]
        resultados.append(
            {
                "grupo": grupo,
                "control": control,
                "estado": estado,
                "detalle": detalle,
                "cantidad": int(cantidad),
                **politica,
            }
        )

    def terminar():
        reporte = pd.DataFrame(resultados)
        aprobado = not reporte.empty and not reporte["estado"].eq("ERROR").any()
        destino = ruta_reporte or config.get("ruta_reporte")
        if destino:
            destino = Path(destino)
            destino.parent.mkdir(parents=True, exist_ok=True)
            reporte.to_csv(destino, index=False, encoding="utf-8")
        return reporte, aprobado

    requeridas_config = {"tipos", "fecha", "inicio", "fin", "clave"}
    faltan_config = sorted(requeridas_config - set(config))
    if faltan_config:
        registrar(
            "Columnas y tipos",
            "Configuracion",
            "ERROR",
            f"Faltan parametros de configuracion: {faltan_config}",
            len(faltan_config),
        )
        return terminar()

    tipos = config["tipos"]
    tipos_validos = {"fecha", "entero", "numerico", "texto"}
    tipos_desconocidos = {
        columna: tipo for columna, tipo in tipos.items() if tipo not in tipos_validos
    }
    if tipos_desconocidos:
        registrar(
            "Columnas y tipos",
            "Configuracion de tipos",
            "ERROR",
            f"Tipos no soportados: {tipos_desconocidos}",
            len(tipos_desconocidos),
        )
        return terminar()

    faltan = sorted(_columnas_referenciadas(config) - set(df.columns))
    if faltan:
        registrar(
            "Columnas y tipos",
            "Columnas indispensables",
            "ERROR",
            f"Faltan columnas requeridas: {faltan}",
            len(faltan),
        )
        return terminar()

    registrar(
        "Columnas y tipos",
        "Columnas indispensables",
        "OK",
        "Todas las columnas requeridas estan presentes.",
    )

    for columna, tipo in tipos.items():
        serie = df[columna]
        if tipo == "fecha":
            incompatible = serie.notna() & pd.to_datetime(serie, errors="coerce").isna()
        elif tipo in ("entero", "numerico"):
            numerica = pd.to_numeric(serie, errors="coerce")
            incompatible = serie.notna() & (numerica.isna() | ~np.isfinite(numerica))
            if tipo == "entero":
                incompatible |= numerica.notna() & np.isfinite(numerica) & (numerica % 1 != 0)
        else:
            incompatible = serie.notna() & ~serie.map(lambda valor: isinstance(valor, str))

        cantidad = int(incompatible.sum())
        registrar(
            "Columnas y tipos",
            f"Tipo: {columna}",
            "ERROR" if cantidad else "OK",
            f"Valores incompatibles con {tipo}: {cantidad}",
            cantidad,
        )

    # Las fechas nulas tambien son invalidas; no se convierten en meses con cero.
    fecha_col = config["fecha"]
    fechas = pd.to_datetime(df[fecha_col], errors="coerce")
    invalidas = fechas.isna()
    registrar(
        "Fechas y cobertura",
        "Fechas validas",
        "ERROR" if invalidas.any() else "OK",
        f"Fechas nulas o invalidas: {int(invalidas.sum())}",
        invalidas.sum(),
    )

    try:
        inicio = pd.Period(config["inicio"], freq="M")
        fin = pd.Period(config["fin"], freq="M")
    except (TypeError, ValueError) as exc:
        registrar(
            "Fechas y cobertura",
            "Periodo configurado",
            "ERROR",
            f"El inicio o fin configurado no es mensual: {exc}",
            1,
        )
        return terminar()

    if inicio > fin:
        registrar(
            "Fechas y cobertura",
            "Periodo configurado",
            "ERROR",
            "El periodo inicial es posterior al periodo final.",
            1,
        )
        return terminar()

    periodos = fechas.dt.to_period("M")
    fuera_periodo = periodos.notna() & ((periodos < inicio) | (periodos > fin))
    registrar(
        "Fechas y cobertura",
        "Registros fuera del periodo",
        "ADVERTENCIA" if fuera_periodo.any() else "OK",
        f"Registros fuera de {inicio} a {fin}: {int(fuera_periodo.sum())}",
        fuera_periodo.sum(),
    )

    esperados = pd.period_range(inicio, fin, freq="M")
    dimensiones = config.get("dimensiones_cobertura", [])
    valores_esperados = config.get("valores_esperados", {})
    valores_por_dimension = [
        valores_esperados.get(columna, df[columna].dropna().unique())
        for columna in dimensiones
    ]
    combinaciones = product(*valores_por_dimension) if dimensiones else [()]
    claves_esperadas = {
        (*combinacion, periodo)
        for combinacion in combinaciones
        for periodo in esperados
    }
    dentro = periodos.notna() & ~fuera_periodo
    claves_observadas = set(
        zip(*(df.loc[dentro, columna] for columna in dimensiones), periodos[dentro])
    )
    faltantes_cobertura = claves_esperadas - claves_observadas
    ejemplos = sorted(faltantes_cobertura, key=str)[:20]
    registrar(
        "Fechas y cobertura",
        "Cobertura mensual",
        "ERROR" if faltantes_cobertura else "OK",
        f"Combinaciones sin cobertura: {ejemplos}",
        len(faltantes_cobertura),
    )

    duplicados = df.duplicated(subset=config["clave"], keep=False)
    registrar(
        "Unicidad",
        "Clave unica",
        "ERROR" if duplicados.any() else "OK",
        f"Filas con claves duplicadas: {int(duplicados.sum())}",
        duplicados.sum(),
    )

    faltantes_permitidos = set(config.get("faltantes_permitidos", []))
    for columna in tipos:
        serie = df[columna]
        faltantes = serie.isna()
        if tipos[columna] == "texto":
            faltantes |= serie.map(lambda valor: isinstance(valor, str) and not valor.strip())
        cantidad = int(faltantes.sum())
        if not cantidad:
            estado = "OK"
        elif columna in faltantes_permitidos:
            estado = "ADVERTENCIA"
        else:
            estado = "ERROR"
        registrar(
            "Valores faltantes",
            f"Faltantes: {columna}",
            estado,
            f"Valores faltantes: {cantidad}; permitidos: {columna in faltantes_permitidos}",
            cantidad,
        )

    for columna in config.get("conteos", []):
        serie = pd.to_numeric(df[columna], errors="coerce")
        malos = serie.isna() | ~np.isfinite(serie) | (serie < 0)
        registrar(
            "Valores y denominadores",
            f"Conteo: {columna}",
            "ERROR" if malos.any() else "OK",
            f"Valores negativos, nulos o no finitos: {int(malos.sum())}",
            malos.sum(),
        )

    for columna in config.get("denominadores", []):
        serie = pd.to_numeric(df[columna], errors="coerce")
        malos = serie.isna() | ~np.isfinite(serie) | (serie <= 0)
        registrar(
            "Valores y denominadores",
            f"Denominador: {columna}",
            "ERROR" if malos.any() else "OK",
            f"Denominadores nulos, no finitos o menores o iguales que cero: {int(malos.sum())}",
            malos.sum(),
        )

    # Una fuente mensual debe representar cada periodo con el primer dia del mes.
    no_mensuales = fechas.notna() & fechas.ne(fechas.dt.to_period("M").dt.to_timestamp())
    registrar(
        "Regla metodologica",
        "Granularidad mensual",
        "ERROR" if no_mensuales.any() else "OK",
        f"Fechas que no representan periodos mensuales: {int(no_mensuales.sum())}",
        no_mensuales.sum(),
    )

    fuente_col = config.get("columna_fuente")
    if fuente_col:
        if "fuente_autorizada" not in config:
            registrar(
                "Regla metodologica",
                "Fuente autorizada",
                "ERROR",
                "Se configuro columna_fuente sin fuente_autorizada.",
                1,
            )
        else:
            incorrectas = df[fuente_col].ne(config["fuente_autorizada"])
            registrar(
                "Regla metodologica",
                "Fuente autorizada",
                "ERROR" if incorrectas.any() else "OK",
                f"Registros de fuente no autorizada: {int(incorrectas.sum())}",
                incorrectas.sum(),
            )

    return terminar()


def validar_o_detener(df, config, ruta_reporte=None):
    """Valida, genera el reporte y detiene el pipeline si hay un error critico."""
    reporte, aprobado = validar_datos(df, config, ruta_reporte=ruta_reporte)
    if not aprobado:
        errores = reporte.loc[reporte["estado"].eq("ERROR"), "control"].tolist()
        raise ErrorCalidadDatos(
            "Los controles de calidad impiden continuar o publicar resultados: "
            + ", ".join(errores)
        )
    return reporte
