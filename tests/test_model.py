import unittest
import sqlite3
import numpy as np

# Se importan las funciones a testear
from src.elo import calcular_resultado_esperado, calcular_nuevo_elo, obtener_elo_equipo
from src.poisson import poisson_probability, predecir_probabilidades_goles, obtener_probabilidades_1X2
from src.utils import obtener_conexion


class TestModeloPredictivo(unittest.TestCase):
    """
    Clase de pruebas unitarias para validar el core matemático y de base de datos
    del proyecto del modelo predictivo de fútbol.
    """

    def test_elo_probabilidad_esperada_equipos_iguales(self):
        """
        Valida que si dos equipos tienen exactamente la misma puntuación ELO,
        la probabilidad esperada de victoria para ambos sea exactamente 0.5 (50%).
        """
        prob_A = calcular_resultado_esperado(1500, 1500)
        prob_B = calcular_resultado_esperado(1500, 1500)
        
        self.assertEqual(prob_A, 0.5)
        self.assertEqual(prob_B, 0.5)

    def test_elo_probabilidad_esperada_equipo_superior(self):
        """
        Valida que un equipo con mayor rating ELO tenga una probabilidad
        esperada de ganar mayor a la de su oponente débil (> 0.5).
        """
        prob_fuerte = calcular_resultado_esperado(1700, 1300)
        prob_debil = calcular_resultado_esperado(1300, 1700)
        
        self.assertGreater(prob_fuerte, 0.5)
        self.assertLess(prob_debil, 0.5)
        # La suma de las probabilidades esperadas ELO debe ser igual a 1
        self.assertAlmostEqual(prob_fuerte + prob_debil, 1.0)

    def test_elo_actualizacion_puntos_victoria(self):
        """
        Valida que tras una victoria (resultado 1.0), el nuevo rating ELO
        sea superior al rating previo del equipo ganador.
        """
        elo_previo = 1500.0
        prob_esperada = 0.5
        nuevo_elo = calcular_nuevo_elo(elo_previo, 1.0, prob_esperada, k=60)
        
        # Debe subir: 1500 + 60 * (1.0 - 0.5) = 1530
        self.assertEqual(nuevo_elo, 1530.0)
        self.assertGreater(nuevo_elo, elo_previo)

    def test_poisson_probabilidad_goles_cero(self):
        """
        Prueba la función de cálculo de Poisson. Si la tasa esperada (lambda)
        de goles es muy baja (casi 0), la probabilidad de anotar 0 goles debe ser cercana a 1.
        """
        prob_cero = poisson_probability(0.0001, 0)
        self.assertAlmostEqual(prob_cero, 1.0, places=3)

    def test_poisson_normalizacion_1X2(self):
        """
        Valida que la suma de las probabilidades individuales de ganar,
        empatar o perder (1X2) sume exactamente 1.0 (100%).
        """
        # Simulamos una predicción ficticia
        xg_A, xg_B, matriz = predecir_probabilidades_goles("Argentina", "France", es_neutral=True, usar_elo=True)
        prob_A, prob_empate, prob_B = obtener_probabilidades_1X2(matriz)
        
        suma_total = prob_A + prob_empate + prob_B
        self.assertAlmostEqual(suma_total, 1.0, places=7)

    def test_conexion_base_de_datos(self):
        """
        Valida que la función de conexión a la base de datos de SQLite
        retorne un objeto de conexión abierto y funcional.
        """
        try:
            conexion = obtener_conexion()
            cursor = conexion.cursor()
            cursor.execute("SELECT 1")
            resultado = cursor.fetchone()
            conexion.close()
            
            # El resultado de SELECT 1 debe ser una tupla con valor 1
            self.assertEqual(resultado[0], 1)
        except sqlite3.Error as e:
            self.fail(f"La conexión a la base de datos falló: {e}")


if __name__ == "__main__":
    unittest.main()
