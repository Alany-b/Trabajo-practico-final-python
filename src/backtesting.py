import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Importamos módulos locales
from src.utils import obtener_conexion
from src.elo import ELO_DEFECTO, VENTAJA_LOCALIA, K_BASE, calcular_k_factor, calcular_resultado_esperado, calcular_nuevo_elo
from src.poisson import poisson_probability


def ejecutar_backtesting_qatar_2022():
    """
    Ejecuta el pipeline de validación temporal (Walk-Forward Validation):
    1. Entrena el modelo ELO e histórico de Poisson con datos hasta antes de Qatar 2022 (Noviembre 2022).
    2. Realiza predicciones sobre los partidos de Qatar 2022.
    3. Evalúa el desempeño usando Log Loss, F1-Score, Accuracy, MAE y RMSE.
    """
    print("="*60)
    print(" INICIANDO PIPELINE DE BACKTESTING: VALIDACIÓN QATAR 2022 ")
    print("="*60)

    conexion = obtener_conexion()
    
    # 1. Cargar todos los partidos
    df_partidos = pd.read_sql_query("""
        SELECT year, tipo_torneo AS stage, home_team AS team1, home_score AS score1,
               away_score AS score2, away_team AS team2, country, date 
        FROM partidos_internacionales 
        ORDER BY year ASC, date ASC
    """, conexion)
    
    conexion.close()

    if df_partidos.empty:
        print("Error: No se encontraron partidos históricos en la base de datos.")
        return

    # Dividimos temporalmente los datos
    # Qatar 2022 se jugó en el año 2022.
    # Usaremos todos los años anteriores y partidos previos a noviembre de 2022 para entrenamiento.
    # El primer partido de Qatar 2022 fue el 2022-11-20 (Ecuador vs Qatar).
    df_entrenamiento = df_partidos[
        (df_partidos['year'] < 2022) | 
        ((df_partidos['year'] == 2022) & (df_partidos['date'] < '2022-11-20'))
    ].copy()
    
    df_prueba = df_partidos[
        (df_partidos['year'] == 2022) & (df_partidos['date'] >= '2022-11-20')
    ].copy()

    print(f"Partidos de Entrenamiento (históricos hasta 2022): {len(df_entrenamiento)}")
    print(f"Partidos de Prueba (Mundial Qatar 2022): {len(df_prueba)}")

    if len(df_prueba) == 0:
        print("Advertencia: No hay partidos del Mundial 2022 en el conjunto de prueba.")
        return

    # --- FASE 1: ENTRENAMIENTO DE RATING ELO ---
    elo_equipos = {}
    
    # Procesamos la evolución de ELO en el dataset de entrenamiento
    # Usamos el K-factor dinámico por año: partidos más recientes tienen mayor peso
    for _, fila in df_entrenamiento.iterrows():
        t1, t2 = fila['team1'], fila['team2']
        score1, score2 = fila['score1'], fila['score2']
        anfitrion = fila['country']
        anio = int(fila['year'])

        if t1 not in elo_equipos:
            elo_equipos[t1] = ELO_DEFECTO
        if t2 not in elo_equipos:
            elo_equipos[t2] = ELO_DEFECTO

        # K-factor dinámico: más peso a partidos recientes (2018-2022)
        k = calcular_k_factor(anio)

        # Ajuste de localía
        r1 = elo_equipos[t1]
        r2 = elo_equipos[t2]

        r1_ajustado = r1 + VENTAJA_LOCALIA if t1 == anfitrion else r1
        r2_ajustado = r2 + VENTAJA_LOCALIA if t2 == anfitrion else r2

        esperado1 = calcular_resultado_esperado(r1_ajustado, r2_ajustado)
        esperado2 = calcular_resultado_esperado(r2_ajustado, r1_ajustado)

        if score1 > score2:
            real1, real2 = 1.0, 0.0
        elif score1 < score2:
            real1, real2 = 0.0, 1.0
        else:
            real1, real2 = 0.5, 0.5

        elo_equipos[t1] = calcular_nuevo_elo(r1, real1, esperado1, k)
        elo_equipos[t2] = calcular_nuevo_elo(r2, real2, esperado2, k)

    # --- FASE 2: ENTRENAMIENTO DE FUERZAS DE POISSON ---
    # Promedio global de goles anotados por equipo en el entrenamiento
    total_goles = df_entrenamiento['score1'].sum() + df_entrenamiento['score2'].sum()
    goles_promedio = total_goles / (2 * len(df_entrenamiento)) if len(df_entrenamiento) > 0 else 1.3

    # Agrupamos estadísticas de goles anotados/recibidos en el entrenamiento
    eq1_goles_anotados = df_entrenamiento.groupby('team1')['score1'].sum()
    eq1_goles_recibidos = df_entrenamiento.groupby('team1')['score2'].sum()
    eq1_partidos = df_entrenamiento.groupby('team1')['score1'].count()

    eq2_goles_anotados = df_entrenamiento.groupby('team2')['score2'].sum()
    eq2_goles_recibidos = df_entrenamiento.groupby('team2')['score1'].sum()
    eq2_partidos = df_entrenamiento.groupby('team2')['score2'].count()

    todos_equipos = set(eq1_partidos.index).union(set(eq2_partidos.index))
    fuerzas = {}

    for eq in todos_equipos:
        pj = eq1_partidos.get(eq, 0) + eq2_partidos.get(eq, 0)
        if pj == 0:
            continue
        g_anotados = eq1_goles_anotados.get(eq, 0) + eq2_goles_anotados.get(eq, 0)
        g_recibidos = eq1_goles_recibidos.get(eq, 0) + eq2_goles_recibidos.get(eq, 0)

        fuerzas[eq] = {
            'ataque': (g_anotados / pj) / goles_promedio if goles_promedio > 0 else 1.0,
            'defensa': (g_recibidos / pj) / goles_promedio if goles_promedio > 0 else 1.0
        }

    # --- FASE 3: PRUEBA Y EVALUACIÓN CON QATAR 2022 ---
    resultados_reales_1X2 = [] # '1' para Local/Equipo1, 'X' para Empate, '2' para Visitante/Equipo2
    resultados_predichos_1X2 = []
    probabilidades_predichas = [] # Lista de arrays de tamaño 3 [prob_1, prob_X, prob_2]
    
    goles_reales_equipo = []
    goles_predichos_equipo = []

    detalles_partidos = []

    # El anfitrión de 2022 fue Qatar
    anfitrion_2022 = "Qatar"

    for _, fila in df_prueba.iterrows():
        t1, t2 = fila['team1'], fila['team2']
        score1, score2 = fila['score1'], fila['score2']

        # 1. Obtener ratings ELO del entrenamiento
        elo_t1 = elo_equipos.get(t1, ELO_DEFECTO)
        elo_t2 = elo_equipos.get(t2, ELO_DEFECTO)

        # 2. Obtener fuerzas Poisson
        f_A = fuerzas.get(t1, {'ataque': 1.0, 'defensa': 1.0})
        f_B = fuerzas.get(t2, {'ataque': 1.0, 'defensa': 1.0})

        # Ajuste localía
        factor_local_ataque = 1.15 if t1 == anfitrion_2022 else 1.0
        factor_local_defensa = 0.90 if t1 == anfitrion_2022 else 1.0

        # Tasa goles iniciales
        lambda1 = f_A['ataque'] * f_B['defensa'] * goles_promedio * factor_local_ataque
        lambda2 = f_B['ataque'] * f_A['defensa'] * goles_promedio * factor_local_defensa

        # Ajuste ELO
        dif_elo = elo_t1 - elo_t2
        lambda1 *= 10.0 ** (dif_elo / 800.0)
        lambda2 *= 10.0 ** (-dif_elo / 800.0)

        # 3. Crear matriz Poisson de goles
        max_goles = 6
        matriz = np.zeros((max_goles, max_goles))
        for x in range(max_goles):
            for y in range(max_goles):
                matriz[x, y] = poisson_probability(lambda1, x) * poisson_probability(lambda2, y)

        # Probabilidades 1X2
        prob_1 = 0.0
        prob_X = 0.0
        prob_2 = 0.0
        for x in range(max_goles):
            for y in range(max_goles):
                if x > y:
                    prob_1 += matriz[x, y]
                elif x < y:
                    prob_2 += matriz[x, y]
                else:
                    prob_X += matriz[x, y]

        # Normalizamos
        tot = prob_1 + prob_X + prob_2
        prob_1, prob_X, prob_2 = prob_1/tot, prob_X/tot, prob_2/tot

        # Predicción final (clase más probable)
        clases_prob = [prob_1, prob_X, prob_2]
        idx_pred = np.argmax(clases_prob)
        clases_map = {0: '1', 1: 'X', 2: '2'}
        pred_1X2 = clases_map[idx_pred]

        # Resultado real
        if score1 > score2:
            real_1X2 = '1'
        elif score1 < score2:
            real_1X2 = '2'
        else:
            real_1X2 = 'X'

        # Guardar para métricas de clasificación
        resultados_reales_1X2.append(real_1X2)
        resultados_predichos_1X2.append(pred_1X2)
        probabilidades_predichas.append(clases_prob)

        # Guardar para métricas de goles
        goles_reales_equipo.extend([score1, score2])
        goles_predichos_equipo.extend([lambda1, lambda2])

        # Detalle individual
        detalles_partidos.append({
            'Partido': f"{t1} vs {t2}",
            'Real': f"{score1}-{score2}",
            'Pred_Marcador': f"{lambda1:.1f} - {lambda2:.1f}",
            'Prob_1X2': f"{prob_1*100:.0f}% / {prob_X*100:.0f}% / {prob_2*100:.0f}%",
            'Acertó': 'Sí' if pred_1X2 == real_1X2 else 'No',
            'Confianza': max(clases_prob)
        })

    # --- FASE 4: CÁLCULO DE MÉTRICAS ---
    # 1. Log Loss para 3 clases (1, X, 2)
    log_losses = []
    for real, probs in zip(resultados_reales_1X2, probabilidades_predichas):
        # probs = [prob_1, prob_X, prob_2]
        if real == '1':
            p_actual = probs[0]
        elif real == 'X':
            p_actual = probs[1]
        else:
            p_actual = probs[2]
        
        # Evitamos log(0)
        p_actual = max(1e-15, min(1 - 1e-15, p_actual))
        log_losses.append(-np.log(p_actual))

    log_loss_promedio = np.mean(log_losses)

    # 2. Accuracy
    acc = accuracy_score(resultados_reales_1X2, resultados_predichos_1X2)

    # 3. F1-Score Macro
    f1 = f1_score(resultados_reales_1X2, resultados_predichos_1X2, average='macro')

    # 4. MAE (Error Medio Absoluto de goles)
    mae = np.mean(np.abs(np.array(goles_reales_equipo) - np.array(goles_predichos_equipo)))

    # 5. RMSE (Error Cuadrático Medio de goles)
    rmse = np.sqrt(np.mean((np.array(goles_reales_equipo) - np.array(goles_predichos_equipo)) ** 2))

    # --- FASE 5: IMPRESIÓN DEL REPORTE FINAL ---
    print("\n" + "="*50)
    print(" METRICAS DE EVALUACIÓN GENERALES (QATAR 2022) ")
    print("="*50)
    print(f"Pérdida Logarítmica (Log Loss):  {log_loss_promedio:.4f}")
    print(f"Exactitud (Accuracy 1X2):        {acc*100:.2f}%")
    print(f"F1-Score Macro (1X2):            {f1:.4f}")
    print(f"Error Medio Absoluto (MAE Goles): {mae:.3f} goles")
    print(f"Raíz Error Cuadrático (RMSE):     {rmse:.3f} goles")
    print("="*50)

    # Convertimos detalles a DataFrame para ver sorpresas
    df_detalles = pd.DataFrame(detalles_partidos)
    print("\nEjemplos de predicciones individuales:")
    print(df_detalles.head(10).to_string(index=False))

    # Mostrar las mayores sorpresas del Mundial (donde el modelo se equivocó y tenía mayor confianza)
    sorpresas = df_detalles[df_detalles['Acertó'] == 'No'].sort_values(by='Confianza', ascending=False)
    print("\nMayores SORPRESAS del torneo (Equivocación con alta confianza):")
    print(sorpresas.head(5).to_string(index=False))


if __name__ == "__main__":
    ejecutar_backtesting_qatar_2022()
