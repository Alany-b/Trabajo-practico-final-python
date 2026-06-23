import os
import sqlite3
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import List

# Importamos nuestros módulos locales
from src.utils import obtener_conexion, clasificar_torneo
from src.elo import procesar_partidos_historicos_y_actualizar_elo, limpiar_cache_elo
from src.poisson import limpiar_cache_poisson
from src.predictor import predecir_partido_completo, ejecutar_monte_carlo, SimuladorMundial
from app.schemas import PredictRequest, PredictResponse, MatchResultInput, TeamInfo, SimulationResponse, PredictMLResponse
import joblib
import pandas as pd

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Fútbol - Mundial 2026",
    description="API REST para predecir partidos de fútbol y simular torneos usando ELO y Poisson.",
    version="1.0.0"
)

# --- CARGA DEL MODELO MACHINE LEARNING (Requisito Académico) ---
modelo_rf = None
estadisticas_ml = None

try:
    if os.path.exists('models/modelo_rf.pkl') and os.path.exists('models/estadisticas.pkl'):
        modelo_rf = joblib.load('models/modelo_rf.pkl')
        estadisticas_ml = joblib.load('models/estadisticas.pkl')
        print("Modelo Machine Learning y Estadísticas cargados correctamente mediante joblib.")
    else:
        print("Advertencia: No se encontraron los archivos .pkl en la carpeta models/.")
except Exception as e:
    print(f"Error al cargar los modelos ML: {e}")



def tarea_segundo_plano_reentrenamiento(resultado: MatchResultInput):
    """
    Función que se ejecuta de forma asíncrona en segundo plano tras registrar
    un nuevo partido. Inserta el partido en SQLite y recalcula ratings.
    """
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        
        # 1. Insertamos el nuevo partido en la tabla de partidos internacionales
        import datetime
        anio_actual = datetime.datetime.now().year
        fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT INTO partidos_internacionales (date, year, home_team, away_team, home_score, away_score, tournament, tipo_torneo, country, neutral)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fecha_actual,
            anio_actual,
            resultado.team_local,
            resultado.team_visitor,
            resultado.goals_local,
            resultado.goals_visitor,
            resultado.competition,
            clasificar_torneo(resultado.competition),
            resultado.team_local,
            1 if resultado.is_neutral else 0
        ))
        
        conexion.commit()
        conexion.close()
        
        # 2. Vaciamos los cachés en memoria para forzar recálculo
        limpiar_cache_elo()
        limpiar_cache_poisson()
        
        # 3. Recalculamos ratings ELO de todos los equipos integrando el nuevo partido
        procesar_partidos_historicos_y_actualizar_elo()
        print("Reentrenamiento del modelo y actualización ELO completados de forma asíncrona.")
        
    except Exception as e:
        print(f"Error en la tarea de segundo plano de reentrenamiento: {e}")


@app.get("/", tags=["General"])
def read_root():
    """
    Ruta raíz para verificar que la API está en línea.
    """
    return {
        "mensaje": "Bienvenido a la API del Modelo Predictivo del Mundial 2026",
        "estado": "En línea",
        "documentacion": "/docs"
    }


@app.get("/teams", response_model=List[TeamInfo], tags=["Equipos"])
def obtener_equipos():
    """
    Retorna la lista de todos los equipos de la base de datos con sus ratings ELO actuales.
    """
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        
        # Hacemos un JOIN para unir la información de equipos con sus ratings ELO calculados
        cursor.execute("""
            SELECT e.team, e.team_group, e.confederation, e.fifa_rank, e.coach, r.rating
            FROM equipos_2026 e
            LEFT JOIN ratings_elo r ON e.team = r.team
            ORDER BY r.rating DESC
        """)
        equipos = cursor.fetchall()
        conexion.close()
        
        return [
            TeamInfo(
                team=eq['team'],
                team_group=eq['team_group'],
                confederation=eq['confederation'],
                fifa_rank=eq['fifa_rank'],
                elo_rating=eq['rating'] if eq['rating'] is not None else 1500.0,
                coach=eq['coach']
            ) for eq in equipos
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {e}")


@app.get("/fixtures", tags=["Calendario"])
def obtener_calendario():
    """
    Retorna el fixture completo de partidos planificados para el Mundial 2026.
    """
    try:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("SELECT * FROM fixtures_2026")
        partidos = cursor.fetchall()
        conexion.close()
        
        return [dict(p) for p in partidos]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {e}")


@app.post("/predict", response_model=PredictResponse, tags=["Predicciones"])
def predecir_partido(req: PredictRequest):
    """
    Calcula la predicción probabilística detallada para un partido entre dos selecciones.
    """
    try:
        prediccion = predecir_partido_completo(
            req.team_local, 
            req.team_visitor, 
            es_neutral=req.is_neutral, 
            usar_elo=req.use_elo
        )
        return PredictResponse(
            team_local=prediccion['equipo_A'],
            team_visitor=prediccion['equipo_B'],
            xg_local=prediccion['xG_A'],
            xg_visitor=prediccion['xG_B'],
            prob_local=prediccion['prob_A'],
            prob_draw=prediccion['prob_empate'],
            prob_visitor=prediccion['prob_B'],
            most_probable_score=prediccion['marcador_probable'],
            prob_score=prediccion['prob_marcador']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al realizar la predicción: {e}")


@app.post("/predict-ml", response_model=PredictMLResponse, tags=["Predicciones ML"])
def predecir_partido_ml(req: PredictRequest):
    """
    Calcula la predicción usando el modelo de Machine Learning (Random Forest)
    exportado desde Scikit-Learn mediante joblib (.pkl).
    """
    if modelo_rf is None or estadisticas_ml is None:
        raise HTTPException(status_code=503, detail="El modelo de Machine Learning no está cargado en el servidor.")
        
    try:
        def obtener_stats(equipo, df_stats):
            row = df_stats[df_stats['equipo'] == equipo]
            if len(row) == 0:
                return {'ataque': 1.0, 'defensa': 1.0}
            return {'ataque': row['ataque'].values[0], 'defensa': row['defensa'].values[0]}

        stats_local = obtener_stats(req.team_local, estadisticas_ml)
        stats_visit = obtener_stats(req.team_visitor, estadisticas_ml)
        
        # Construimos el dataframe con los features esperados por el modelo
        X_input = pd.DataFrame([{
            'ataque_local': stats_local['ataque'],
            'defensa_local': stats_local['defensa'],
            'ataque_visit': stats_visit['ataque'],
            'defensa_visit': stats_visit['defensa'],
            'es_neutral': 1 if req.is_neutral else 0,
            'peso_torneo': 4 # Asumimos Mundial
        }])
        
        # predict_proba retorna probabilidades para las clases [0 (Visitante), 1 (Empate), 2 (Local)]
        proba = modelo_rf.predict_proba(X_input)[0]
        clases_modelo = modelo_rf.classes_ # Generalmente [0, 1, 2]
        
        prob_v = proba[list(clases_modelo).index(0)] if 0 in clases_modelo else 0.0
        prob_e = proba[list(clases_modelo).index(1)] if 1 in clases_modelo else 0.0
        prob_l = proba[list(clases_modelo).index(2)] if 2 in clases_modelo else 0.0
        
        resultado_clase = modelo_rf.predict(X_input)[0]
        if resultado_clase == 2:
            res_str = "Gana Local"
        elif resultado_clase == 1:
            res_str = "Empate"
        else:
            res_str = "Gana Visitante"
            
        return PredictMLResponse(
            team_local=req.team_local,
            team_visitor=req.team_visitor,
            prob_local=prob_l,
            prob_draw=prob_e,
            prob_visitor=prob_v,
            predicted_result=res_str
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al realizar la predicción ML: {e}")


@app.post("/update-match", tags=["Reentrenamiento"])
def registrar_resultado_partido(resultado: MatchResultInput, tareas: BackgroundTasks):
    """
    Registra el resultado real de un partido.
    Dispara de forma asíncrona (segundo plano) la actualización del modelo y ELO, 
    permitiendo que la API responda inmediatamente.
    """
    # Encolamos la tarea en segundo plano usando las utilidades de FastAPI
    tareas.add_task(tarea_segundo_plano_reentrenamiento, resultado)
    
    return {
        "estado": "Recibido",
        "mensaje": "El resultado ha sido encolado para registro y reentrenamiento asíncrono del modelo."
    }


@app.post("/simulate-tournament", response_model=SimulationResponse, tags=["Simulación"])
def simular_torneo_completo():
    """
    Simula una única vez la fase completa de eliminación directa del mundial
    y retorna la estructura de campeones y semifinalistas resultantes.
    """
    try:
        simulador = SimuladorMundial()
        res, _ = simulador.ejecutar_simulacion_torneo()
        simulador.cerrar_conexion()
        return SimulationResponse(
            campeon=res['campeon'],
            subcampeon=res['subcampeon'],
            tercero=res['tercero'],
            cuarto=res['cuarto'],
            semifinalistas=res['semifinalistas'],
            cuartofinalistas=res['cuartofinalistas']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al simular el torneo: {e}")


@app.get("/simulate-montecarlo", tags=["Simulación"])
def simular_montecarlo(iteraciones: int = 100):
    """
    Ejecuta una simulación masiva de Monte Carlo para calcular los porcentajes
    de éxito de cada selección en las distintas fases del campeonato.
    """
    if iteraciones < 1 or iteraciones > 1000:
        raise HTTPException(status_code=400, detail="El número de iteraciones debe ser entre 1 y 1000.")
    try:
        df_resultados = ejecutar_monte_carlo(iteraciones=iteraciones)
        # Convertimos el DataFrame a diccionario JSON para la respuesta
        return df_resultados.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la simulación Monte Carlo: {e}")
