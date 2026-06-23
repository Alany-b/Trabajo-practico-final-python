import os
import sqlite3
import pandas as pd
from src.utils import obtener_conexion

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
VENTAJA_LOCALIA = 60    # Puntos ELO de bonus para el equipo local
ELO_DEFECTO     = 1500  # Rating inicial para selecciones sin historial
K_BASE          = 40    # K-factor base sobre el que se aplican multiplicadores


# ─────────────────────────────────────────────────────────────────────────────
# K-FACTOR DINÁMICO (por tipo de torneo × recencia)
# ─────────────────────────────────────────────────────────────────────────────
# Pesos por categoría de torneo
K_PESOS_TORNEO = {
    'mundial'          : 2.5,   # Copa del Mundo: máximo impacto
    'eliminatoria'     : 1.5,   # Clasificatorias: alta relevancia
    'copa_continental' : 1.8,   # Copa América, Euro, Copa África...
    'amistoso'         : 0.5,   # Amistosos: bajo peso
    'otro'             : 0.6,   # Torneos menores
}

def calcular_k_factor(anio, tipo_torneo='amistoso'):
    """
    Calcula el K-factor dinámico de un partido según dos dimensiones:

    1. TIPO DE TORNEO → determina la importancia del resultado:
       - Mundial:          K_BASE × 2.5  (máximo impacto)
       - Copa Continental: K_BASE × 1.8
       - Eliminatoria:     K_BASE × 1.5
       - Amistoso:         K_BASE × 0.5  (mínimo impacto)

    2. RECENCIA → los resultados más cercanos al 2026 pesan más:
       - 2010-2013: × 0.7   (base del período moderno)
       - 2014-2017: × 0.85  (levemente reducido)
       - 2018-2021: × 1.1   (alta relevancia: ciclo previo al 2022)
       - 2022-2026: × 1.4   (máximo: Qatar 2022 + preparación 2026)

    Parámetros:
    - anio        : Año del partido (int)
    - tipo_torneo : Categoría del torneo (str)

    Retorna:
    - K-factor calculado (float)
    """
    peso_torneo = K_PESOS_TORNEO.get(tipo_torneo, 0.6)

    if anio <= 2013:
        peso_anio = 0.70
    elif anio <= 2017:
        peso_anio = 0.85
    elif anio <= 2021:
        peso_anio = 1.10
    else:
        peso_anio = 1.40   # 2022–2026: máxima relevancia (Qatar + preparación)

    return K_BASE * peso_torneo * peso_anio


# ─────────────────────────────────────────────────────────────────────────────
# FÓRMULA ELO ESTÁNDAR
# ─────────────────────────────────────────────────────────────────────────────
def calcular_resultado_esperado(rating_A, rating_B):
    """
    Calcula la probabilidad esperada de victoria del Equipo A según la
    fórmula matemática estándar del sistema ELO.

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Parámetros:
    - rating_A: Rating ELO del equipo A
    - rating_B: Rating ELO del equipo B

    Retorna float entre 0 y 1.
    """
    return 1.0 / (10.0 ** ((rating_B - rating_A) / 400.0) + 1.0)


def calcular_nuevo_elo(rating_actual, resultado_real, resultado_esperado, k):
    """
    Aplica la actualización de rating ELO tras un partido.

    R' = R + K × (S - E)
      R : Rating anterior
      K : K-factor dinámico del partido
      S : Resultado real  (1.0 victoria, 0.5 empate, 0.0 derrota)
      E : Probabilidad esperada (calculada antes del partido)

    Parámetros:
    - rating_actual     : Rating ELO anterior al partido
    - resultado_real    : 1.0 / 0.5 / 0.0
    - resultado_esperado: Probabilidad calculada
    - k                 : K-factor del partido

    Retorna el nuevo rating (float).
    """
    return rating_actual + k * (resultado_real - resultado_esperado)


# ─────────────────────────────────────────────────────────────────────────────
# ELO INICIAL BASADO EN RANKING FIFA
# ─────────────────────────────────────────────────────────────────────────────
def elo_desde_ranking_fifa(fifa_rank):
    """
    Estima un ELO de arranque realista basado en el ranking FIFA actual.

    Lógica empírica calibrada:
    - Top 5 FIFA  → ~1750-1780 pts ELO
    - Top 20 FIFA → ~1650-1700 pts ELO
    - Top 50 FIFA → ~1580-1620 pts ELO
    - >100 FIFA   → ~1450-1500 pts ELO
    - Piso mínimo : 1300 pts (para debutantes muy débiles)

    Parámetros:
    - fifa_rank: Posición en el ranking FIFA (int)

    Retorna float.
    """
    elo = 1800 - (fifa_rank * 4.5)
    return max(1300.0, round(elo, 2))


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO HISTÓRICO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def procesar_partidos_historicos_y_actualizar_elo():
    """
    Procesa todos los partidos internacionales desde 2010 para calcular
    el rating ELO actual de cada selección mediante K-factor dinámico.

    Pipeline:
    1. Inicializar ELO de los 48 equipos clasificados al 2026 usando su
       ranking FIFA actual (referencia objetiva y actualizada).
    2. Recorrer todos los partidos en orden cronológico desde 2010.
    3. Actualizar ELO de cada equipo con K ajustado por torneo y recencia.
    4. Guardar los ratings finales en la tabla 'ratings_elo' de SQLite.
    """
    conexion = obtener_conexion()
    cursor   = conexion.cursor()

    # ── 1. Cargar equipos clasificados al 2026 con su ranking FIFA ──
    cursor.execute("SELECT team, confederation, fifa_rank FROM equipos_2026")
    equipos_2026 = cursor.fetchall()

    # Inicializar ELO de todos los equipos del 2026 desde su ranking FIFA
    # Esto asegura que Argentina, Francia, Brasil empiecen con ELO alto
    # y Ecuador con uno acorde a su ranking real (#24 → ~1692).
    elo_equipos = {}
    for eq in equipos_2026:
        elo_equipos[eq['team']] = elo_desde_ranking_fifa(eq['fifa_rank'])

    # ── 2. Cargar todos los partidos internacionales desde 2010 ──
    df = pd.read_sql_query("""
        SELECT date, year, home_team, away_team,
               home_score, away_score, tipo_torneo, country, neutral
        FROM partidos_internacionales
        ORDER BY date ASC
    """, conexion)

    if df.empty:
        print("Error: No hay partidos en partidos_internacionales. Ejecutá src.utils primero.")
        conexion.close()
        return

    print(f"Procesando {len(df)} partidos internacionales (2010–2026)...")

    # ── 3. Recorrer partidos cronológicamente y actualizar ELO ──
    for _, fila in df.iterrows():
        home     = fila['home_team']
        away     = fila['away_team']
        g_home   = fila['home_score']
        g_away   = fila['away_score']
        anio     = int(fila['year'])
        torneo   = fila['tipo_torneo']
        neutral  = bool(fila['neutral'])

        # Inicializar equipos que aparecen en el historial pero no en 2026
        if home not in elo_equipos:
            elo_equipos[home] = ELO_DEFECTO
        if away not in elo_equipos:
            elo_equipos[away] = ELO_DEFECTO

        # K-factor dinámico (torneo × recencia)
        k = calcular_k_factor(anio, torneo)

        # Aplicar ventaja de localía solo si no es cancha neutral
        r_home = elo_equipos[home]
        r_away = elo_equipos[away]

        if not neutral:
            r_home_adj = r_home + VENTAJA_LOCALIA
            r_away_adj = r_away
        else:
            r_home_adj = r_home
            r_away_adj = r_away

        # Probabilidades esperadas según ELO actual
        esp_home = calcular_resultado_esperado(r_home_adj, r_away_adj)
        esp_away = calcular_resultado_esperado(r_away_adj, r_home_adj)

        # Resultado real del partido
        if g_home > g_away:
            real_home, real_away = 1.0, 0.0
        elif g_home < g_away:
            real_home, real_away = 0.0, 1.0
        else:
            real_home, real_away = 0.5, 0.5

        # Actualizar ratings con K dinámico
        elo_equipos[home] = calcular_nuevo_elo(r_home, real_home, esp_home, k)
        elo_equipos[away] = calcular_nuevo_elo(r_away, real_away, esp_away, k)

    # ── 4. Guardar ratings en SQLite ──
    cursor.execute("DELETE FROM ratings_elo")

    equipos_meta = {eq['team']: eq for eq in equipos_2026}

    for equipo, rating in elo_equipos.items():
        meta = equipos_meta.get(equipo)
        conf = meta['confederation'] if meta else 'Desconocido'
        rank = meta['fifa_rank']     if meta else None

        cursor.execute("""
            INSERT OR REPLACE INTO ratings_elo (team, rating, confederation, fifa_rank)
            VALUES (?, ?, ?, ?)
        """, (equipo, round(rating, 2), conf, rank))

    conexion.commit()
    conexion.close()
    print(f"[OK] Ratings ELO actualizados. Selecciones procesadas: {len(elo_equipos)}")


# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ EN MEMORIA (para simulaciones Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────
_elo_cache = {}

def limpiar_cache_elo():
    """Vacía el caché de ELO para forzar recarga desde SQLite."""
    global _elo_cache
    _elo_cache.clear()

def obtener_elo_equipo(equipo, usar_cache=True):
    """
    Retorna el rating ELO de una selección desde la base de datos.
    Usa caché en RAM para evitar queries repetitivas durante simulaciones.

    Parámetros:
    - equipo     : Nombre del equipo (str)
    - usar_cache : Si True, usa caché en RAM (recomendado para simulaciones)
    """
    global _elo_cache

    if usar_cache and equipo in _elo_cache:
        return _elo_cache[equipo]

    conexion = obtener_conexion()
    cursor   = conexion.cursor()
    cursor.execute("SELECT rating FROM ratings_elo WHERE team = ?", (equipo,))
    resultado = cursor.fetchone()
    conexion.close()

    rating = resultado['rating'] if resultado else ELO_DEFECTO

    if usar_cache:
        _elo_cache[equipo] = rating

    return rating


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    procesar_partidos_historicos_y_actualizar_elo()
