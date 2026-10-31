"""Demuestra que una fuente defectuosa no reemplaza la version aprobada."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import guardar_json, leer_config


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Prueba controlada de columna ausente')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config_fallo.yaml')))
    args = parser.parse_args()
    config, ruta_config = leer_config(args.config)
    puntero = Path(config['rutas']['version_vigente'])
    if not puntero.is_file():
        raise FileNotFoundError('Se necesita una version aprobada antes de ejecutar la prueba')

    contenido_antes = puntero.read_bytes()
    version_antes = json.loads(contenido_antes)
    aprobada_antes = Path(config['rutas']['resultados_aprobados']) / version_antes['id_ejecucion'] / 'estimaciones.csv'
    hash_aprobada_antes = huella(aprobada_antes)
    reportes = Path(config['rutas']['reportes'])
    carpetas_antes = {ruta.name for ruta in reportes.iterdir() if ruta.is_dir()}

    comando = [sys.executable, str(Path(__file__).with_name('pipeline.py')),
               '--config', str(ruta_config)]
    proceso = subprocess.run(comando, capture_output=True, text=True, check=False)
    carpetas_despues = {ruta.name for ruta in reportes.iterdir() if ruta.is_dir()}
    nuevas = sorted(c for c in carpetas_despues - carpetas_antes
                    if (reportes / c / 'ejecucion.json').is_file())
    if len(nuevas) != 1:
        raise RuntimeError(f'No se identifico exactamente una ejecucion de prueba: {nuevas}')

    with (reportes / nuevas[0] / 'ejecucion.json').open(encoding='utf-8') as archivo:
        ejecucion = json.load(archivo)
    contenido_despues = puntero.read_bytes()
    version_despues = json.loads(contenido_despues)
    hash_aprobada_despues = huella(aprobada_antes)
    error_columna = any('Columnas originales faltantes' in error
                        for error in ejecucion.get('errores', []))
    aprobado = (
        proceso.returncode != 0
        and ejecucion.get('estado') == 'fallido'
        and error_columna
        and version_antes['id_ejecucion'] == version_despues['id_ejecucion']
        and contenido_antes == contenido_despues
        and hash_aprobada_antes == hash_aprobada_despues
    )
    evidencia = {
        'fecha': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'prueba': 'columna Viajero ausente',
        'aprobada': aprobado,
        'codigo_salida_pipeline': proceso.returncode,
        'ejecucion_fallida': nuevas[0],
        'estado_ejecucion': ejecucion.get('estado'),
        'error_columna_detectado': error_columna,
        'version_antes': version_antes['id_ejecucion'],
        'version_despues': version_despues['id_ejecucion'],
        'puntero_sin_cambios': contenido_antes == contenido_despues,
        'estimaciones_aprobadas_sin_cambios': hash_aprobada_antes == hash_aprobada_despues,
        'stdout': proceso.stdout[-2000:],
        'stderr': proceso.stderr[-2000:],
    }
    destino = reportes / 'pruebas_controladas' / f'fallo_{nuevas[0]}.json'
    guardar_json(evidencia, destino)
    print(json.dumps(evidencia, indent=2, ensure_ascii=False))
    return 0 if aprobado else 1


if __name__ == '__main__':
    sys.exit(main())
