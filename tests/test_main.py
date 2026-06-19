import math
import unittest

from pydantic import ValidationError

import main


def evaluacion_valida(**changes):
    data = {
        "peso_kg": 60,
        "talla_m": 1.7,
        "imc": 10,  # El servidor debe ignorar este valor y recalcularlo.
        "sum_3_pl": 35,
        "sum_6_pl": 70,
        "perimetro_cintura": 75,
        "perimetro_brazo_relajado": 27,
        "perimetro_muslo": 50,
        "perimetro_pantorrilla": 35,
        "diametro_humero": 6,
        "diametro_femur": 9,
        "horas_sueno": 8,
        "carga_entrenamiento": 450,
    }
    data.update(changes)
    return data


class ValidationTests(unittest.TestCase):
    def test_imc_se_calcula_en_servidor(self):
        evaluation = main.EvaluacionNueva(**evaluacion_valida())
        self.assertTrue(math.isclose(evaluation.imc, 20.76))

    def test_rechaza_medidas_fuera_de_rango(self):
        for field, value in [
            ("peso_kg", -1), ("talla_m", 0),
            ("horas_sueno", 25), ("carga_entrenamiento", -1),
        ]:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                main.EvaluacionNueva(**evaluacion_valida(**{field: value}))

    def test_rechaza_jugador_invalido(self):
        with self.assertRaises(ValidationError):
            main.JugadorNuevo(nombre="1", edad=3, posicion="Portero")

    def test_interfaz_usa_api_del_mismo_origen(self):
        html = main.cargar_interfaz()
        self.assertIn('const API_URL = "/api"', html)
        self.assertIn("function escapeHtml", html)


if __name__ == "__main__":
    unittest.main()
