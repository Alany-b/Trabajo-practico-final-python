import os
import sqlite3
import random
import numpy as np
import pandas as pd
from src.utils import obtener_conexion
from src.poisson import predecir_probabilidades_goles, obtener_probabilidades_1X2


def predecir_partido_completo(equipo_A, equipo_B, es_neutral=True, usar_elo=True):
    """
    Predice las estadísticas completas de un partido entre el Equipo A y el Equipo B.
    
    Retorna:
    - Un diccionario con goles esperados (xG), probabilidades 1X2 y el marcador más probable.
    """
    # Obtenemos las fuerzas de Poisson y la matriz de probabilidad de goles
    xg_A, xg_B, matriz = predecir_probabilidades_goles(equipo_A, equipo_B, es_neutral, usar_elo)
    
    # Obtenemos las probabilidades de victoria A, empate y victoria B
    prob_A, prob_empate, prob_B = obtener_probabilidades_1X2(matriz)
    
    # Encontrar el marcador más probable (el de mayor valor en la matriz 6x6)
    idx_max = np.unravel_index(np.argmax(matriz), matriz.shape)
    goles_A_probable = int(idx_max[0])
    goles_B_probable = int(idx_max[1])
    prob_marcador = float(matriz[idx_max])

    return {
        'equipo_A': equipo_A,
        'equipo_B': equipo_B,
        'xG_A': float(xg_A),
        'xG_B': float(xg_B),
        'prob_A': float(prob_A),
        'prob_empate': float(prob_empate),
        'prob_B': float(prob_B),
        'marcador_probable': (goles_A_probable, goles_B_probable),
        'prob_marcador': prob_marcador
    }


def simular_marcador_partido(lambda_A, lambda_B, es_eliminacion_directa=False):
    """
    Simula el marcador de un partido usando distribuciones de Poisson independientes.
    Si es un partido de eliminación directa y termina empatado, simula prórroga y penaltis.
    
    Retorna:
    - goles_A, goles_B, ganador (nombre del equipo 'A' o 'B')
    """
    # Generamos la cantidad de goles en base al xG (distribución de Poisson)
    goles_A = np.random.poisson(lambda_A)
    goles_B = np.random.poisson(lambda_B)

    if not es_eliminacion_directa:
        # En fase de grupos se permiten empates
        if goles_A > goles_B:
            ganador = 'A'
        elif goles_A < goles_B:
            ganador = 'B'
        else:
            ganador = 'Empate'
        return goles_A, goles_B, ganador

    # Para eliminación directa no puede haber empates
    if goles_A > goles_B:
        return goles_A, goles_B, 'A'
    elif goles_B > goles_A:
        return goles_A, goles_B, 'B'
    else:
        # 1. Simulación de Prórroga (30 minutos extras)
        # El xG disminuye proporcionalmente a la duración (30% del tiempo de un partido regular)
        goles_extra_A = np.random.poisson(lambda_A * 0.3)
        goles_extra_B = np.random.poisson(lambda_B * 0.3)
        
        total_goles_A = goles_A + goles_extra_A
        total_goles_B = goles_B + goles_extra_B

        if total_goles_A > total_goles_B:
            return total_goles_A, total_goles_B, 'A'
        elif total_goles_B > total_goles_A:
            return total_goles_A, total_goles_B, 'B'
        else:
            # 2. Simulación de Tanda de Penales (50% de probabilidad para cada uno)
            # En penales se le suma un gol técnico al ganador oficial
            ganador = 'A' if random.random() < 0.5 else 'B'
            if ganador == 'A':
                return total_goles_A + 1, total_goles_B, 'A'
            else:
                return total_goles_A, total_goles_B + 1, 'B'


class SimuladorMundial:
    """
    Clase encargada de estructurar y simular todas las etapas del Mundial 2026.
    Integra la fase de grupos de 48 equipos y los cruces de eliminación directa.
    """
    def __init__(self):
        self.conexion = obtener_conexion()
        self.cargar_datos_iniciales()

    def cargar_datos_iniciales(self):
        """
        Carga la información de equipos clasificados y el calendario desde SQLite.
        """
        cursor = self.conexion.cursor()
        
        # Cargar los equipos
        cursor.execute("SELECT team, team_group, fifa_rank FROM equipos_2026")
        self.equipos = [dict(row) for row in cursor.fetchall()]
        
        # Agrupar equipos por su grupo original (A-L)
        self.grupos = {}
        for eq in self.equipos:
            g = eq['team_group']
            if g not in self.grupos:
                self.grupos[g] = []
            self.grupos[g].append(eq['team'])

        # Cargar los partidos del calendario oficial
        cursor.execute("SELECT * FROM fixtures_2026")
        self.fixtures = [dict(row) for row in cursor.fetchall()]

    def simular_fase_grupos(self):
        """
        Simula los partidos de la fase de grupos y calcula la tabla de posiciones final.
        
        Retorna:
        - standings: Diccionario con la tabla de posiciones de cada grupo.
        - clasificados_directos: Lista de los 2 primeros de cada grupo.
        - mejores_terceros: Lista de los 8 mejores terceros clasificados.
        """
        # Inicializamos las estadísticas de los 48 equipos
        stats = {
            eq['team']: {
                'equipo': eq['team'],
                'grupo': eq['team_group'],
                'PJ': 0, 'PG': 0, 'PE': 0, 'PP': 0,
                'GF': 0, 'GC': 0, 'DG': 0, 'Pts': 0
            } for eq in self.equipos
        }

        # Filtramos partidos de fase de grupos
        partidos_grupos = [f for f in self.fixtures if f['stage'] == 'Group Stage']

        for part in partidos_grupos:
            eq1 = part['team1']
            eq2 = part['team2']

            # Obtenemos los xG esperados de Poisson
            xg1, xg2, _ = predecir_probabilidades_goles(eq1, eq2, es_neutral=True, usar_elo=True)
            
            # Simulamos el partido
            goles1, goles2, ganador = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=False)

            # Actualizamos las estadísticas de ambos equipos
            stats[eq1]['PJ'] += 1
            stats[eq2]['PJ'] += 1
            stats[eq1]['GF'] += goles1
            stats[eq1]['GC'] += goles2
            stats[eq2]['GF'] += goles2
            stats[eq2]['GC'] += goles1
            stats[eq1]['DG'] = stats[eq1]['GF'] - stats[eq1]['GC']
            stats[eq2]['DG'] = stats[eq2]['GF'] - stats[eq2]['GC']

            if ganador == 'A':
                stats[eq1]['PG'] += 1
                stats[eq1]['Pts'] += 3
                stats[eq2]['PP'] += 1
            elif ganador == 'B':
                stats[eq2]['PG'] += 1
                stats[eq2]['Pts'] += 3
                stats[eq1]['PP'] += 1
            else:
                stats[eq1]['PE'] += 1
                stats[eq2]['PE'] += 1
                stats[eq1]['Pts'] += 1
                stats[eq2]['Pts'] += 1

        # Ordenar tablas de posiciones de cada grupo
        # Criterio: Puntos (Pts) -> Diferencia Goles (DG) -> Goles Favor (GF) -> Orden aleatorio
        tablas_grupos = {}
        for g_nombre, g_equipos in self.grupos.items():
            equipos_stats = [stats[eq] for eq in g_equipos]
            # Ordenamos de mayor a menor utilizando una clave de ordenamiento estructurada
            equipos_ordenados = sorted(
                equipos_stats, 
                key=lambda x: (x['Pts'], x['DG'], x['GF']), 
                reverse=True
            )
            tablas_grupos[g_nombre] = equipos_ordenados

        # Seleccionar clasificados directos (1° y 2° de cada grupo de la A a la L)
        clasificados_directos = {}
        terceros = []

        for g_nombre, tabla in tablas_grupos.items():
            # Los dos primeros clasifican directamente
            clasificados_directos[f"1{g_nombre}"] = tabla[0]['equipo']
            clasificados_directos[f"2{g_nombre}"] = tabla[1]['equipo']
            # El tercero va al pool de terceros para evaluar
            terceros.append(tabla[2])

        # Ordenamos los terceros lugares para obtener los 8 mejores
        terceros_ordenados = sorted(
            terceros, 
            key=lambda x: (x['Pts'], x['DG'], x['GF']), 
            reverse=True
        )

        # Los mejores 8 terceros clasifican y se etiquetan como Best 3rd #1 a #8
        mejores_terceros = {}
        for i in range(8):
            mejores_terceros[f"Best 3rd #{i+1}"] = terceros_ordenados[i]['equipo']

        return tablas_grupos, clasificados_directos, mejores_terceros

    def simular_fases_eliminatorias(self, clasificados_directos, mejores_terceros):
        """
        Simula toda la ronda de eliminación directa del Mundial (ronda de 32 a la final).
        Mapea dinámicamente los ganadores a las llaves de la ronda siguiente.
        """
        # --- 1. RONDA DE 32 ---
        partidos_r32 = [f for f in self.fixtures if f['stage'] == 'Round of 32']
        ganadores_r32 = {}

        for i, part in enumerate(partidos_r32, start=1):
            t1_placeholder = part['team1']
            t2_placeholder = part['team2']

            # Mapeamos los placeholders (ej. '1A', 'Best 3rd #1') a los equipos reales
            equipo1 = clasificados_directos.get(t1_placeholder) or mejores_terceros.get(t1_placeholder)
            equipo2 = clasificados_directos.get(t2_placeholder) or mejores_terceros.get(t2_placeholder)

            # Si alguno es nulo (por seguridad en la lógica), asignamos un placeholder
            equipo1 = equipo1 or "Equipo Desconocido 1"
            equipo2 = equipo2 or "Equipo Desconocido 2"

            xg1, xg2, _ = predecir_probabilidades_goles(equipo1, equipo2, es_neutral=True, usar_elo=True)
            _, _, ganador = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)

            equipo_ganador = equipo1 if ganador == 'A' else equipo2
            ganadores_r32[f"R32 W{i}"] = equipo_ganador

        # --- 2. OCTAVOS DE FINAL (ROUND OF 16) ---
        partidos_r16 = [f for f in self.fixtures if f['stage'] == 'Round of 16']
        ganadores_r16 = {}

        for i, part in enumerate(partidos_r16, start=1):
            t1_placeholder = part['team1']
            t2_placeholder = part['team2']

            equipo1 = ganadores_r32[t1_placeholder]
            equipo2 = ganadores_r32[t2_placeholder]

            xg1, xg2, _ = predecir_probabilidades_goles(equipo1, equipo2, es_neutral=True, usar_elo=True)
            _, _, ganador = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)

            equipo_ganador = equipo1 if ganador == 'A' else equipo2
            ganadores_r16[f"R16 W{i}"] = equipo_ganador

        # --- 3. CUARTOS DE FINAL ---
        partidos_qf = [f for f in self.fixtures if f['stage'] == 'Quarter-final']
        ganadores_qf = {}

        for i, part in enumerate(partidos_qf, start=1):
            t1_placeholder = part['team1']
            t2_placeholder = part['team2']

            # En la DB los equipos en Cuartos de final están etiquetados como 'QF1', 'QF2', etc.
            # Los mapeamos a los ganadores de octavos ('R16 W1', 'R16 W2', etc.)
            idx1 = int(t1_placeholder.replace("QF", ""))
            idx2 = int(t2_placeholder.replace("QF", ""))

            equipo1 = ganadores_r16[f"R16 W{idx1}"]
            equipo2 = ganadores_r16[f"R16 W{idx2}"]

            xg1, xg2, _ = predecir_probabilidades_goles(equipo1, equipo2, es_neutral=True, usar_elo=True)
            _, _, ganador = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)

            equipo_ganador = equipo1 if ganador == 'A' else equipo2
            ganadores_qf[f"QF W{i}"] = equipo_ganador

        # --- 4. SEMIFINALES ---
        partidos_sf = [f for f in self.fixtures if f['stage'] == 'Semi-final']
        ganadores_sf = {}
        perdedores_sf = {}

        for i, part in enumerate(partidos_sf, start=1):
            t1_placeholder = part['team1']
            t2_placeholder = part['team2']

            # En la DB los equipos en Semifinales están etiquetados como 'SF1', 'SF2', etc.
            # Los mapeamos a los ganadores de cuartos ('QF W1', 'QF W2', etc.)
            idx1 = int(t1_placeholder.replace("SF", ""))
            idx2 = int(t2_placeholder.replace("SF", ""))

            equipo1 = ganadores_qf[f"QF W{idx1}"]
            equipo2 = ganadores_qf[f"QF W{idx2}"]

            xg1, xg2, _ = predecir_probabilidades_goles(equipo1, equipo2, es_neutral=True, usar_elo=True)
            _, _, ganador = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)

            if ganador == 'A':
                ganadores_sf[f"SF W{i}"] = equipo1
                perdedores_sf[f"SF L{i}"] = equipo2
            else:
                ganadores_sf[f"SF W{i}"] = equipo2
                perdedores_sf[f"SF L{i}"] = equipo1

        # --- 5. TERCER PUESTO (3rd Place Match) ---
        partido_3pl = [f for f in self.fixtures if f['stage'] == '3rd Place Match'][0]
        equipo1_3 = perdedores_sf['SF L1']
        equipo2_3 = perdedores_sf['SF L2']

        xg1, xg2, _ = predecir_probabilidades_goles(equipo1_3, equipo2_3, es_neutral=True, usar_elo=True)
        _, _, ganador_3 = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)
        tercer_lugar = equipo1_3 if ganador_3 == 'A' else equipo2_3
        cuarto_lugar = equipo2_3 if ganador_3 == 'A' else equipo1_3

        # --- 6. FINAL ---
        partido_f = [f for f in self.fixtures if f['stage'] == 'Final'][0]
        equipo1_f = ganadores_sf['SF W1']
        equipo2_f = ganadores_sf['SF W2']

        xg1, xg2, _ = predecir_probabilidades_goles(equipo1_f, equipo2_f, es_neutral=True, usar_elo=True)
        _, _, ganador_f = simular_marcador_partido(xg1, xg2, es_eliminacion_directa=True)
        campeon = equipo1_f if ganador_f == 'A' else equipo2_f
        subcampeon = equipo2_f if ganador_f == 'A' else equipo1_f

        return {
            'campeon': campeon,
            'subcampeon': subcampeon,
            'tercero': tercer_lugar,
            'cuarto': cuarto_lugar,
            'semifinalistas': list(ganadores_qf.values()),
            'cuartofinalistas': list(ganadores_r16.values())
        }

    def ejecutar_simulacion_torneo(self):
        """
        Ejecuta una simulación completa del torneo (Fase de grupos + Eliminatoria).
        """
        tablas, directos, terceros = self.simular_fase_grupos()
        resultados_fase_final = self.simular_fases_eliminatorias(directos, terceros)
        return resultados_fase_final, tablas

    def cerrar_conexion(self):
        self.conexion.close()


def ejecutar_monte_carlo(iteraciones=1000):
    """
    Ejecuta múltiples simulaciones del torneo para estimar las probabilidades
    porcentuales de cada equipo en las diferentes fases del Mundial.
    
    Retorna:
    - Un DataFrame de Pandas con las probabilidades estimadas para cada selección.
    """
    simulador = SimuladorMundial()

    # Contadores de éxito para cada equipo
    campeonatos = {}
    finales = {}
    semis = {}
    cuartos = {}
    clasificados_grupo = {}

    for i in range(iteraciones):
        res, tablas = simulador.ejecutar_simulacion_torneo()

        # 1. Registrar clasificados de grupo
        # Analizamos quiénes quedaron 1° y 2° en la simulación
        for grupo, tabla in tablas.items():
            for eq_pos, eq_data in enumerate(tabla):
                eq = eq_data['equipo']
                # Si está en los dos primeros puestos
                if eq_pos < 2:
                    clasificados_grupo[eq] = clasificados_grupo.get(eq, 0) + 1

        # 2. Registrar fases finales
        camp = res['campeon']
        sub = res['subcampeon']
        t3 = res['tercero']
        t4 = res['cuarto']

        campeonatos[camp] = campeonatos.get(camp, 0) + 1

        finales[camp] = finales.get(camp, 0) + 1
        finales[sub] = finales.get(sub, 0) + 1

        semis[camp] = semis.get(camp, 0) + 1
        semis[sub] = semis.get(sub, 0) + 1
        semis[t3] = semis.get(t3, 0) + 1
        semis[t4] = semis.get(t4, 0) + 1

        # Cuartofinalistas
        for eq in res['semifinalistas']:
            cuartos[eq] = cuartos.get(eq, 0) + 1

    simulador.cerrar_conexion()

    # Consolidar los resultados en una tabla estructurada
    datos_consolidados = []
    for eq_dict in simulador.equipos:
        eq = eq_dict['team']
        
        datos_consolidados.append({
            'Equipo': eq,
            'Grupo': eq_dict['team_group'],
            'Clasifica_Grupo_%': (clasificados_grupo.get(eq, 0) / iteraciones) * 100,
            'Llega_Cuartos_%': (cuartos.get(eq, 0) / iteraciones) * 100,
            'Llega_Semis_%': (semis.get(eq, 0) / iteraciones) * 100,
            'Llega_Final_%': (finales.get(eq, 0) / iteraciones) * 100,
            'Campeon_%': (campeonatos.get(eq, 0) / iteraciones) * 100
        })

    df_resultado = pd.DataFrame(datos_consolidados)
    # Ordenamos por probabilidad de campeonar
    df_resultado = df_resultado.sort_values(by='Campeon_%', ascending=False).reset_index(drop=True)
    return df_resultado


if __name__ == "__main__":
    # Ejecuta una simulación de Monte Carlo rápida (100 iteraciones)
    print("Iniciando simulación de Monte Carlo del Mundial 2026...")
    df_probabilidades = ejecutar_monte_carlo(iteraciones=100)
    print("\nTop 10 equipos con más probabilidades de ser Campeón del Mundo 2026:")
    print(df_probabilidades[['Equipo', 'Clasifica_Grupo_%', 'Campeon_%']].head(10))
