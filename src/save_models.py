import os
import pickle
from src.utils import obtener_conexion
from src.elo import obtener_elo_equipo
from src.poisson import calcular_fuerzas_ataque_defensa

# Rutas de guardado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def serializar_y_guardar_modelos():
    """
    Calcula los parámetros actuales de los modelos (ELO y Poisson) 
    y los guarda como archivos binarios pickle (.pkl) en la carpeta 'models/'.
    """
    print("Guardando modelos entrenados en la carpeta 'models/'...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Obtener y guardar fuerzas de Poisson
    fuerzas, goles_promedio = calcular_fuerzas_ataque_defensa()
    poisson_data = {
        "fuerzas": fuerzas,
        "goles_promedio": goles_promedio
    }
    poisson_path = os.path.join(MODELS_DIR, "poisson_model.pkl")
    with open(poisson_path, "wb") as f:
        pickle.dump(poisson_data, f)
    print(f"- Modelo de Poisson guardado en: {poisson_path}")

    # 2. Obtener y guardar todos los ratings ELO calculados
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("SELECT team, rating FROM ratings_elo")
    elos = {row['team']: row['rating'] for row in cursor.fetchall()}
    conexion.close()

    elo_path = os.path.join(MODELS_DIR, "elo_ratings.pkl")
    with open(elo_path, "wb") as f:
        pickle.dump(elos, f)
    print(f"- Ratings ELO serializados en: {elo_path}")
    print("Modelos guardados con éxito.")


if __name__ == "__main__":
    serializar_y_guardar_modelos()
