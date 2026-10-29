"""Preparación reproducible de datos INGUAT (primera etapa del pipeline).

Ejecutar desde cualquier directorio:
    python pipeline.py --config config.yaml

Lee datos/Datos_INGUAT.csv usando las rutas relativas al config.yaml,
limpia las columnas indicadas, agrega viajeros por vía y mes y divide
el conjunto en 70%/30% de los meses disponibles, sin mezclar fechas.
No entrena modelos todavía.
"""

import argparse
import hashlib
import json
import logging
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


def normalizar_texto(valor: object) -> str:
    """Minúsculas, sin acentos y sin espacios sobrantes."""
    texto = str(valor).strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def cargar_configuracion(ruta_config: Path) -> dict:
    with ruta_config.open("r", encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)
    proporcion = float(config["particion"]["entrenamiento"])
    proporcion_val = float(config["particion"]["validacion"])
    if not (0 < proporcion < 1) or abs(proporcion + proporcion_val - 1) > 1e-9:
        raise ValueError("Las proporciones de entrenamiento y validación deben sumar 1.")
    if config["particion"]["criterio"] != "primeros_meses_con_datos":
        raise ValueError("Se espera una partición cronológica por primeros meses con datos.")
    if config["particion"]["dividir_por"] != "meses_unicos":
        raise ValueError("La partición debe efectuarse por meses únicos, no por filas.")
    return config


def cargar_datos(config: dict, carpeta_base: Path) -> tuple[pd.DataFrame, Path]:
    ruta = (carpeta_base / config["datos"]["ruta"]).resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")

    # Detecta separadores comunes (coma, punto y coma, tabulador).
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    requeridas = list(config["datos"]["columnas"].values())
    faltantes = sorted(set(requeridas).difference(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")

    return df[requeridas].copy(), ruta


def limpiar_datos(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepara registros fuente antes de agrupar; no convierte ausencias a cero."""
    columnas = config["datos"]["columnas"]
    df = df.rename(columns={nombre: clave for clave, nombre in columnas.items()})

    # Comprobar tipos y valores antes de agregarlos.
    for nombre in ("anio", "mes", "viajeros"):
        df[nombre] = pd.to_numeric(df[nombre], errors="coerce")
        if df[nombre].isna().any():
            raise ValueError(
                f"La columna '{columnas[nombre]}' contiene valores vacíos o no numéricos."
            )

    if ((df["anio"] % 1 != 0) | (df["mes"] % 1 != 0)).any():
        raise ValueError("Año y Mes cod deben ser números enteros.")
    if ((df["mes"] < 1) | (df["mes"] > 12)).any():
        raise ValueError("Mes cod debe contener meses de 1 a 12.")
    if ((df["viajeros"] < 0) | (~df["viajeros"].map(pd.notna))).any():
        raise ValueError("Viajero no puede contener conteos negativos o faltantes.")
    if not df["viajeros"].map(lambda x: x != float("inf") and x != float("-inf")).all():
        raise ValueError("Viajero contiene valores infinitos.")

    decimales = (df["viajeros"] % 1 != 0)

    if decimales.any():
        logging.warning("Se detectaron %d registros decimales en Viajero. Ejemplos: %s", int(decimales.sum()), df.loc[decimales, "viajeros"].head(5).tolist())
    if df[["via", "tipo_viajero"]].isna().any().any():
        raise ValueError("Faltan valores de Vía o Tipo de Viajero.")

    df["fecha"] = pd.to_datetime(
        {"year": df["anio"].astype(int),
         "month": df["mes"].astype(int),
         "day": 1},
        errors="coerce",
    )
    if df["fecha"].isna().any():
        raise ValueError("Hay combinaciones Año/Mes cod inválidas.")

    # Las variantes con tilde y plural se normalizan únicamente para comparar.
    alias_tipos = {
        "turista": "turista", "turistas": "turista",
        "excursionista": "excursionista", "excursionistas": "excursionista",
    }
    df["tipo_viajero"] = df["tipo_viajero"].map(normalizar_texto).replace(alias_tipos)
    permitidos = {normalizar_texto(x) for x in config["datos"]["tipos_viajero"]}
    omitidas = int((~df["tipo_viajero"].isin(permitidos)).sum())
    if omitidas:
        logging.warning("Se excluyen %d filas de tipos de viajero fuera del alcance.", omitidas)
    df = df.loc[df["tipo_viajero"].isin(permitidos)].copy()
    if df.empty:
        raise ValueError("No hay turistas/excursionistas con los valores configurados.")

    alias_vias = {
        "aerea": "aereo", "aereo": "aereo",
        "maritima": "maritimo", "maritimo": "maritimo",
        "terrestre": "terrestre",
    }
    df["via"] = df["via"].map(normalizar_texto).replace(alias_vias)
    vias_esperadas = set(config["salidas"]["archivos"])
    otras_vias = sorted(set(df["via"]) - vias_esperadas)
    if otras_vias:
        raise ValueError(f"Vías sin archivo de modelo configurado: {otras_vias}")
    return df


def preparar_series(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por vía y mes; País/Tipo de Viajero ya se integran por suma."""
    mensual = (
        df.groupby(["via", "fecha"], as_index=False, sort=True)["viajeros"]
        .sum()
        .sort_values(["via", "fecha"])
        .reset_index(drop=True)
    )
    return mensual


def particion_temporal(mensual: pd.DataFrame, config: dict):
    """Corta por meses presentes en el conjunto, nunca por número de filas."""
    meses = pd.DatetimeIndex(sorted(mensual["fecha"].unique()))
    corte = int(len(meses) * float(config["particion"]["entrenamiento"]))
    if corte == 0 or corte == len(meses):
        raise ValueError("Se necesitan suficientes meses distintos para separar 70/30.")

    meses_train = meses[:corte]
    meses_validacion = meses[corte:]
    train = mensual[mensual["fecha"].isin(meses_train)].copy()
    validacion = mensual[mensual["fecha"].isin(meses_validacion)].copy()
    return train, validacion, meses


def meses_faltantes(meses: pd.DatetimeIndex) -> list[str]:
    """Identifica huecos temporales, sin rellenarlos artificialmente."""
    if len(meses) == 0:
        return []
    todos = pd.date_range(meses.min(), meses.max(), freq="MS")
    return [fecha.strftime("%Y-%m") for fecha in todos.difference(meses)]


def ejecutar(ruta_config: Path) -> None:
    ruta_config = ruta_config.resolve()
    config = cargar_configuracion(ruta_config)
    carpeta_base = ruta_config.parent
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df, ruta_datos = cargar_datos(config, carpeta_base)
    df_limpio = limpiar_datos(df, config)
    mensual = preparar_series(df_limpio)
    train, validacion, meses = particion_temporal(mensual, config)

    destino = carpeta_base / "resultados" / "preparacion"
    destino.mkdir(parents=True, exist_ok=True)
    mensual.to_csv(destino / "series_mensuales.csv", index=False)
    train.to_csv(destino / "entrenamiento.csv", index=False)
    validacion.to_csv(destino / "validacion.csv", index=False)

    huecos_por_via = {
        via: meses_faltantes(pd.DatetimeIndex(grupo["fecha"].unique()))
        for via, grupo in mensual.groupby("via")
    }
    reporte = {
        "ruta_datos": str(ruta_datos),
        "sha256_datos": hashlib.sha256(ruta_datos.read_bytes()).hexdigest(),
        "meses_con_datos": len(meses),
        "meses_entrenamiento": train["fecha"].nunique(),
        "meses_validacion": validacion["fecha"].nunique(),
        "inicio_entrenamiento": train["fecha"].min().strftime("%Y-%m"),
        "fin_entrenamiento": train["fecha"].max().strftime("%Y-%m"),
        "inicio_validacion": validacion["fecha"].min().strftime("%Y-%m"),
        "fin_validacion": validacion["fecha"].max().strftime("%Y-%m"),
        "vias": sorted(mensual["via"].unique().tolist()),
        "meses_sin_cobertura_global": meses_faltantes(meses),
        "meses_sin_cobertura_por_via": huecos_por_via,
    }
    (destino / "reporte_preparacion.json").write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logging.info("Meses disponibles: %d", len(meses))
    logging.info("Entrenamiento: %s a %s (%d meses)",
                 reporte["inicio_entrenamiento"], reporte["fin_entrenamiento"],
                 reporte["meses_entrenamiento"])
    logging.info("Validación: %s a %s (%d meses)",
                 reporte["inicio_validacion"], reporte["fin_validacion"],
                 reporte["meses_validacion"])
    if reporte["meses_sin_cobertura_global"] or any(huecos_por_via.values()):
        logging.warning("Se detectaron meses sin cobertura; no se reemplazaron por ceros.")
    logging.info("Archivos de preparación: %s", destino)
    logging.info("ETAPA COMPLETADA: lectura, limpieza, agregación y partición. Sin ajustar modelos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparación de datos INGUAT")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    argumentos = parser.parse_args()
    ejecutar(argumentos.config)
