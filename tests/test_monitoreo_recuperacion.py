import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from pipeline import guardar_json, hash_archivo, monitorear, restaurar


class MonitoreoRecuperacionTest(unittest.TestCase):
    def config_base(self, raiz):
        return {
            'rutas': {
                'resultados_aprobados': str(raiz / 'aprobados'),
                'version_vigente': str(raiz / 'version_vigente.json'),
                'reportes': str(raiz / 'reportes'),
                'modelos': str(raiz / 'modelos'),
            },
            'controles': {
                'final': {'cobertura': {'fecha_fin': '2026-06'}},
            },
            'modelos': {
                'vias': ['AEREA', 'TERRESTRE'],
                'pronostico': {'fecha_inicio': '2026-07', 'horizonte_meses': 2},
            },
            'monitoreo': {
                'variacion_relativa_maxima': 0.5,
                'umbral_duplicados': 0,
                'umbral_faltantes_criticos': 0,
                'alertar_actualizacion_atrasada': True,
            },
        }

    def test_monitoreo_alerta_sin_rechazar_variacion(self):
        with tempfile.TemporaryDirectory() as temporal:
            config = self.config_base(Path(temporal))
            limpio = pd.DataFrame({'fecha': pd.to_datetime(['2026-05-01'])})
            predicciones = pd.DataFrame({
                'via': ['AEREA', 'AEREA', 'TERRESTRE', 'TERRESTRE'],
                'fecha': pd.to_datetime(['2026-07-01', '2026-08-01'] * 2),
                'estimacion': [100.0, 200.0, 100.0, 110.0],
            })
            informe = {
                'controles': {'final': {'detalle': {
                    'duplicados': 0, 'nulos_fecha': 0,
                    'cobertura_AEREA': {'faltantes_criticos': 0},
                    'cobertura_TERRESTRE': {'faltantes_criticos': 0},
                }}},
                'modelos': {
                    'AEREA': {'meses_evaluados': ['2025-01']},
                    'TERRESTRE': {'meses_evaluados': ['2025-01']},
                },
                'advertencias': [],
            }

            monitorear(predicciones, limpio, config, informe)

            self.assertEqual(informe['monitoreo']['verificaciones']['actualizacion']['estado'], 'ALERTA')
            self.assertEqual(informe['monitoreo']['verificaciones']['resultados']['estado'], 'ALERTA')
            self.assertTrue(informe['monitoreo']['error_predictivo_actualizado'])
            self.assertTrue(any('requiere revision' in alerta
                                for alerta in informe['monitoreo']['alertas']))

    def test_restauracion_verifica_y_cambia_puntero(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            config = self.config_base(raiz)
            identificador = '20261031T123304Z_888cb1b5'
            carpeta = raiz / 'aprobados' / identificador
            carpeta.mkdir(parents=True)
            estimaciones = pd.DataFrame({
                'via': ['AEREA', 'AEREA', 'TERRESTRE', 'TERRESTRE'],
                'fecha': ['2026-07-01', '2026-08-01'] * 2,
                'estimacion': [100.0, 110.0, 200.0, 210.0],
            })
            estimaciones.to_csv(carpeta / 'estimaciones.csv', index=False)
            guardar_json({
                'id_ejecucion': identificador,
                'mlflow_run_id': 'run-prueba',
                'sha256_estimaciones': hash_archivo(carpeta / 'estimaciones.csv'),
                'aceptacion': {'aprobado': True},
            }, carpeta / 'metadatos.json')
            for via in ('aerea', 'terrestre'):
                modelo = raiz / 'modelos' / identificador / f'{via}.joblib'
                modelo.parent.mkdir(parents=True, exist_ok=True)
                modelo.write_bytes(b'modelo-prueba')

            puntero = restaurar(config, identificador)

            self.assertEqual(puntero['id_ejecucion'], identificador)
            with (raiz / 'version_vigente.json').open(encoding='utf-8') as archivo:
                self.assertEqual(json.load(archivo)['accion'], 'restauracion')
            evidencias = list((raiz / 'reportes' / 'restauraciones').glob('*.json'))
            self.assertEqual(len(evidencias), 1)


if __name__ == '__main__':
    unittest.main()
