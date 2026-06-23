import os
import math
import sqlite3
import pandas as pd
import numpy as np
from src.utils import obtener_conexion
from src.elo import obtener_elo_equipo


def calcular_fuerzas_ataque_defensa():
    """
    Analiza la tabla 'partidos_internacionales' en SQLite para calcular
    las fuerzas de ataque y defensa de cada selección nacional.

    Usa el dataset completo (2010+) con más de 12.000 partidos internacionales,
    lo que da estimaciones mucho más representativas del nivel actual de cada
    selección que el antiguo dataset limitado de solo mundiales.

    Retorna:
    - Un diccionario {equipo: {'ataque': float, 'defensa': float}}
    - El promedio de goles por equipo por partido (lambda base para Poisson)
    """
    conexion = obtener_conexion()

    # Usamos la tabla con el dataset internacional completo (2010+)
    # Renombramos columnas para mantener compatibilidad con el resto del código
    df = pd.read_sql_query("""
        SELECT home_team AS team1, home_score AS score1,
               away_score AS score2, away_team AS team2
        FROM partidos_internacionales
    """, conexion)
    conexion.close()

    # Si la tabla está vacía, devolvemos valores neutros por defecto
    if df.empty:
        return {}, 1.3

    # Promedio global de goles anotados por equipo por partido
    total_goles_anotados = df['score1'].sum() + df['score2'].sum()
    total_partidos       = len(df)
    goles_promedio = total_goles_anotados / (2 * total_partidos) if total_partidos > 0 else 1.3

    fuerzas = {}

    # Estadísticas cuando el equipo juega como local (home_team)
    eq1_goles_anotados  = df.groupby('team1')['score1'].sum()
    eq1_goles_recibidos = df.groupby('team1')['score2'].sum()
    eq1_partidos        = df.groupby('team1')['score1'].count()

    # Estadísticas cuando el equipo juega como visitante (away_team)
    eq2_goles_anotados  = df.groupby('team2')['score2'].sum()
    eq2_goles_recibidos = df.groupby('team2')['score1'].sum()
    eq2_partidos        = df.groupby('team2')['score2'].count()

    # Combinamos estadísticas de local y visitante para cada selección
    todos_los_equipos = set(eq1_partidos.index).union(set(eq2_partidos.index))

    for equipo in todos_los_equipos:
        # Sumamos partidos jugados
        partidos_jugados = eq1_partidos.get(equipo, 0) + eq2_partidos.get(equipo, 0)
        if partidos_jugados == 0:
            continue

        # Sumamos goles anotados y recibidos
        goles_anotados = eq1_goles_anotados.get(equipo, 0) + eq2_goles_anotados.get(equipo, 0)
        goles_recibidos = eq1_goles_recibidos.get(equipo, 0) + eq2_goles_recibidos.get(equipo, 0)

        # Promedios del equipo
        promedio_anotados = goles_anotados / partidos_jugados
        promedio_recibidos = goles_recibidos / partidos_jugados

        # Fuerza de ataque = Promedio anotados del equipo / promedio general de goles
        # Un valor > 1 indica un ataque mejor que el promedio
        fuerza_ataque = promedio_anotados / goles_promedio if goles_promedio > 0 else 1.0

        # Fuerza de defensa = Promedio recibidos del equipo / promedio general de goles
        # Un valor < 1 indica una defensa mejor que el promedio (recibe menos goles)
        fuerza_defensa = promedio_recibidos / goles_promedio if goles_promedio > 0 else 1.0

        fuerzas[equipo] = {
            'ataque': fuerza_ataque,
            'defensa': fuerza_defensa,
            'partidos': partidos_jugados
        }

    return fuerzas, goles_promedio


def poisson_probability(lambda_param, k):
    """
    Calcula la probabilidad de que ocurran exactamente k eventos
    según una distribución de Poisson con parámetro lambda.
    
    Fórmula: P(X=k) = (lambda^k * e^-lambda) / k!
    """
    if lambda_param <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lambda_param) * (lambda_param ** k)) / math.factorial(k)


# Cache global en memoria para evitar repetir consultas y cálculos pesados en simulaciones
_fuerzas_cache = None
_goles_promedio_cache = None


def limpiar_cache_poisson():
    """
    Limpia el caché en memoria de las fuerzas y goles promedio.
    """
    global _fuerzas_cache, _goles_promedio_cache
    _fuerzas_cache = None
    _goles_promedio_cache = None


def predecir_probabilidades_goles(equipo_A, equipo_B, es_neutral=True, usar_elo=True, fuerzas=None, goles_promedio=None):
    """
    Calcula la matriz de probabilidad de goles entre el Equipo A y el Equipo B
    utilizando el modelo de Poisson y ajustando por su rating ELO.
    
    Parámetros:
    - equipo_A: Nombre de la primera selección.
    - equipo_B: Nombre de la segunda selección.
    - es_neutral: Indica si juegan en cancha neutral.
    - usar_elo: Si es True, ajusta las tasas de goles basándose en la diferencia ELO.
    - fuerzas: Opcional. Diccionario de fuerzas ya calculado (para optimizar).
    - goles_promedio: Opcional. Promedio de goles ya calculado (para optimizar).
    
    Retorna:
    - lambda_A: Goles esperados (xG) para el Equipo A.
    - lambda_B: Goles esperados (xG) para el Equipo B.
    - matriz_probabilidades: Matriz NumPy (6x6) con probabilidades de marcadores exactos.
    """
    global _fuerzas_cache, _goles_promedio_cache

    # Si no se proveen las fuerzas, intentamos obtenerlas desde el caché o recalcularlas
    if fuerzas is None or goles_promedio is None:
        if _fuerzas_cache is None or _goles_promedio_cache is None:
            _fuerzas_cache, _goles_promedio_cache = calcular_fuerzas_ataque_defensa()
        fuerzas = _fuerzas_cache
        goles_promedio = _goles_promedio_cache

    # Si un equipo no tiene historial de goles, le asignamos valores promedio (fuerza = 1.0)
    datos_A = fuerzas.get(equipo_A, {'ataque': 1.0, 'defensa': 1.0})
    datos_B = fuerzas.get(equipo_B, {'ataque': 1.0, 'defensa': 1.0})

    # Ajuste de localía: Si no es neutral, asumimos que Equipo A es local
    # La localía típicamente incrementa el ataque local y reduce su defensa
    factor_local_ataque = 1.15 if not es_neutral else 1.0
    factor_local_defensa = 0.90 if not es_neutral else 1.0

    # Tasa básica de goles para cada equipo (lambda y mu)
    lambda_A = datos_A['ataque'] * datos_B['defensa'] * goles_promedio * factor_local_ataque
    lambda_B = datos_B['ataque'] * datos_A['defensa'] * goles_promedio * factor_local_defensa

    # Ajuste ELO: Modifica los goles esperados según la diferencia de calidad actual
    if usar_elo:
        elo_A = obtener_elo_equipo(equipo_A)
        elo_B = obtener_elo_equipo(equipo_B)
        
        # Diferencia de ratings ELO
        dif_elo = elo_A - elo_B
        
        # Multiplicador exponencial basado en ELO (escala de 400 puntos de diferencia duplica la probabilidad)
        multiplicador_A = 10.0 ** (dif_elo / 800.0)
        multiplicador_B = 10.0 ** (-dif_elo / 800.0)
        
        lambda_A *= multiplicador_A
        lambda_B *= multiplicador_B

    # Creamos la matriz de probabilidades de 0 a 5 goles por equipo (6x6)
    max_goles = 6
    matriz_probabilidades = np.zeros((max_goles, max_goles))

    for x in range(max_goles):
        for y in range(max_goles):
            prob_x = poisson_probability(lambda_A, x)
            prob_y = poisson_probability(lambda_B, y)
            matriz_probabilidades[x, y] = prob_x * prob_y

    return lambda_A, lambda_B, matriz_probabilidades


def obtener_probabilidades_1X2(matriz):
    """
    Suma las probabilidades de la matriz para obtener los resultados de
    Victoria Local/Equipo A (1), Empate (X) y Victoria Visitante/Equipo B (2).
    """
    prob_A = 0.0  # Victoria Equipo A (x > y)
    prob_empate = 0.0  # Empate (x == y)
    prob_B = 0.0  # Victoria Equipo B (x < y)

    max_goles = matriz.shape[0]

    for x in range(max_goles):
        for y in range(max_goles):
            if x > y:
                prob_A += matriz[x, y]
            elif x < y:
                prob_B += matriz[x, y]
            else:
                prob_empate += matriz[x, y]

    # Normalizamos para que la suma sea exactamente 1.0 ante posibles pérdidas de precisión decimal
    total = prob_A + prob_empate + prob_B
    if total > 0:
        prob_A /= total
        prob_empate /= total
        prob_B /= total

    return prob_A, prob_empate, prob_B


if __name__ == "__main__":
    # Prueba rápida del modelo de Poisson
    eq1, eq2 = "Argentina", "France"
    xg_A, xg_B, matriz = predecir_probabilidades_goles(eq1, eq2, es_neutral=True)
    p1, px, p2 = obtener_probabilidades_1X2(matriz)
    
    print(f"Predicción: {eq1} vs {eq2}")
    print(f"Goles Esperados (xG): {eq1} {xg_A:.2f} - {xg_B:.2f} {eq2}")
    print(f"Probabilidades: {eq1}: {p1*100:.1f}% | Empate: {px*100:.1f}% | {eq2}: {p2*100:.1f}%")
