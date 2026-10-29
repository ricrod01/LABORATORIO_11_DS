import pandas as pd
import numpy as np
from itertools import product


def validar_datos(df, config):

    resultados = []

    def registrar(control, estado, detalle, cantidad=0):
        resultados.append({
            "control": control,
            "estado": estado,
            "detalle": detalle,
            "cantidad": cantidad
        })

    # A. COLUMNAS Y TIPOS
    tipos = config["tipos"]

    faltan = set(tipos) - set(df.columns)

    if faltan:
        registrar(
            "Columnas y tipos", "ERROR",
            f"Faltan: {sorted(faltan)}",
            len(faltan)
        )
        return pd.DataFrame(resultados), False

    for columna, tipo in tipos.items():
        s = df[columna]

        if tipo == "fecha":
            invalido = (
                s.notna()
                & pd.to_datetime(s, errors="coerce").isna()
            )

        elif tipo in ("entero", "numerico"):
            num = pd.to_numeric(s, errors="coerce")
            invalido = s.notna() & num.isna()

            if tipo == "entero":
                invalido |= (
                    num.notna()
                    & np.isfinite(num)
                    & (num % 1 != 0)
                )

        elif tipo == "texto":
            invalido = s.notna() & ~s.map(
                lambda x: isinstance(x, str)
            )

        else:
            raise ValueError(f"Tipo no soportado: {tipo}")

        registrar(
            f"Tipo: {columna}",
            "ERROR" if invalido.any() else "OK",
            f"Valores incompatibles: {invalido.sum()}",
            int(invalido.sum())
        )

    # B. FECHAS Y COBERTURA
    fecha_col = config["fecha"]
    fechas = pd.to_datetime(
        df[fecha_col], errors="coerce"
    )

    invalidas = fechas.isna()

    registrar(
        "Fechas válidas",
        "ERROR" if invalidas.any() else "OK",
        f"Fechas nulas o inválidas: {invalidas.sum()}",
        int(invalidas.sum())
    )

    if not invalidas.any():
        periodos = fechas.dt.to_period("M")

        inicio = pd.Period(config["inicio"], freq="M")
        fin = pd.Period(config["fin"], freq="M")

        esperados = pd.period_range(inicio, fin, freq="M")

        dimensiones = config["dimensiones_cobertura"]

        combinaciones = [
            config["valores_esperados"][c]
            for c in dimensiones
        ]

        claves_esperadas = set(
            product(*combinaciones, esperados)
        )

        claves_observadas = set(
            zip(
                *(df[c] for c in dimensiones),
                periodos
            )
        )

        faltantes = claves_esperadas - claves_observadas

        registrar(
            "Cobertura mensual",
            "ERROR" if faltantes else "OK",
            f"Combinaciones sin cobertura: "
            f"{sorted(faltantes, key=str)[:20]}",
            len(faltantes)
        )

    # C. UNICIDAD
    duplicados = df.duplicated(
        subset=config["clave"],
        keep=False
    )

    registrar(
        "Unicidad",
        "ERROR" if duplicados.any() else "OK",
        f"Filas con claves duplicadas: {duplicados.sum()}",
        int(duplicados.sum())
    )

    # D. VALORES FALTANTES
    for columna in tipos:

        n = int(df[columna].isna().sum())

        if n == 0:
            estado = "OK"
        elif columna in config["faltantes_permitidos"]:
            estado = "ADVERTENCIA"
        else:
            estado = "ERROR"

        registrar(
            f"Faltantes: {columna}",
            estado,
            f"{n} valores faltantes",
            n
        )

    # E. VALORES Y DENOMINADORES
    for columna in config["conteos"]:
        s = pd.to_numeric(df[columna], errors="coerce")

        malos = (~np.isfinite(s)) | (s < 0)

        registrar(
            f"Conteo: {columna}",
            "ERROR" if malos.any() else "OK",
            f"Valores negativos o no finitos: {malos.sum()}",
            int(malos.sum())
        )

    for columna in config["denominadores"]:
        s = pd.to_numeric(df[columna], errors="coerce")

        malos = (~np.isfinite(s)) | (s <= 0)

        registrar(
            f"Denominador: {columna}",
            "ERROR" if malos.any() else "OK",
            f"Denominadores inválidos: {malos.sum()}",
            int(malos.sum())
        )

    # F. REGLA METODOLÓGICA
    fuente_col = config["columna_fuente"]
    fuente_esperada = config["fuente_autorizada"]

    incorrectas = df[fuente_col] != fuente_esperada

    registrar(
        "Fuente metodológica",
        "ERROR" if incorrectas.any() else "OK",
        f"Registros de fuente no autorizada: "
        f"{incorrectas.sum()}",
        int(incorrectas.sum())
    )

    reporte = pd.DataFrame(resultados)

    aprobado = not reporte["estado"].eq("ERROR").any()

    return reporte, aprobado