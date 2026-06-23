import os
import sqlite3
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(BASE_DIR, "datasets", "processed", "futbol_predictivo.db")
RAW_DATA_DIR = os.path.join(BASE_DIR, "datasets", "raw")

# Nombres de los archivos CSV de entrada
CSV_INTERNACIONAL = "selecciones-partidos-completos.csv"   # Dataset principal (2010+)
CSV_TEAMS_2026    = "wc_2026_teams.csv"
CSV_FIXTURES_2026 = "wc_2026_fixtures.csv"


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES DE EQUIPOS
# ─────────────────────────────────────────────────────────────────────────────
# Mapa de sinónimos: convierte nombres históricos/alternativos al nombre oficial
# que usa el dataset wc_2026_teams.csv (referencia del Mundial 2026).
NORMALIZACION_EQUIPOS = {
    # Nombres alternativos en el dataset internacional
    'United States'             : 'USA',
    'IR Iran'                   : 'Iran',
    'Korea Republic'            : 'South Korea',
    'Korea DPR'                 : 'North Korea',
    "Côte d'Ivoire"             : 'Ivory Coast',
    'Bosnia and Herzegovina'    : 'Bosnia-Herzegovina',
    'Türkiye'                   : 'Turkey',
    'Kyrgyz Republic'           : 'Kyrgyzstan',
    'São Tomé and Príncipe'     : 'Sao Tome and Principe',
    'Cape Verde Islands'        : 'Cape Verde',
    # Nombres históricos en wc_all_matches (legado)
    'West Germany'              : 'Germany',
    'Soviet Union'              : 'Russia',
    'Zaire'                     : 'DR Congo',
    'Dutch East Indies'         : 'Indonesia',
    'Czechoslovakia'            : 'Czech Republic',
}


def normalizar_nombre(nombre):
    """Retorna el nombre normalizado de una selección si existe en el mapa."""
    return NORMALIZACION_EQUIPOS.get(str(nombre).strip(), str(nombre).strip())


# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN A LA BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def obtener_conexion():
    """
    Establece y retorna una conexión a la base de datos SQLite.
    Configura row_factory para acceder a columnas por nombre (como diccionario).
    """
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conexion = sqlite3.connect(DB_PATH)
        conexion.row_factory = sqlite3.Row
        return conexion
    except sqlite3.Error as e:
        print(f"Error al conectar con la base de datos: {e}")
        raise e


# ─────────────────────────────────────────────────────────────────────────────
# CREACIÓN DE TABLAS
# ─────────────────────────────────────────────────────────────────────────────
def inicializar_base_de_datos():
    """
    Crea todas las tablas necesarias en SQLite si no existen aún.
    """
    conexion = obtener_conexion()
    cursor   = conexion.cursor()

    # Tabla principal: partidos internacionales desde 2010 (nuevo dataset)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS partidos_internacionales (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT,
            year         INTEGER,
            home_team    TEXT,
            away_team    TEXT,
            home_score   INTEGER,
            away_score   INTEGER,
            tournament   TEXT,
            tipo_torneo  TEXT,
            country      TEXT,
            neutral      INTEGER
        );
    """)

    # Selecciones clasificadas al Mundial 2026
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equipos_2026 (
            team          TEXT PRIMARY KEY,
            team_group    TEXT,
            confederation TEXT,
            fifa_rank     INTEGER,
            coach         TEXT,
            best_wc_result TEXT,
            debut_2026    TEXT
        );
    """)

    # Calendario de partidos del Mundial 2026
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fixtures_2026 (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            team_group           TEXT,
            stage                TEXT,
            team1                TEXT,
            team2                TEXT,
            venue                TEXT,
            city                 TEXT,
            country              TEXT,
            date                 TEXT,
            kickoff_et           TEXT,
            team1_confederation  TEXT,
            team1_fifa_rank      INTEGER,
            team1_coach          TEXT,
            team2_confederation  TEXT,
            team2_fifa_rank      INTEGER,
            team2_coach          TEXT
        );
    """)

    # Tabla de ratings ELO calculados
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings_elo (
            team          TEXT PRIMARY KEY,
            rating        REAL,
            confederation TEXT,
            fifa_rank     INTEGER
        );
    """)

    conexion.commit()
    conexion.close()
    print("Tablas creadas (o ya existían).")


# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICACIÓN DE TORNEOS
# ─────────────────────────────────────────────────────────────────────────────
def clasificar_torneo(nombre_torneo):
    """
    Clasifica un torneo en una de 4 categorías según su importancia.

    Categorías:
    - 'mundial'          → Partidos de la Copa del Mundo FIFA (máxima importancia)
    - 'eliminatoria'     → Clasificatorias para el Mundial
    - 'copa_continental' → Copas América, UEFA Euro, Copa África, Gold Cup, etc.
    - 'amistoso'         → Partidos amistosos internacionales (menor importancia)
    - 'otro'             → Torneos regionales menores
    """
    nombre = str(nombre_torneo).lower()

    if 'world cup' in nombre and 'qualif' not in nombre:
        return 'mundial'
    elif 'qualif' in nombre or 'eliminatoria' in nombre:
        return 'eliminatoria'
    elif any(x in nombre for x in [
        'copa america', 'euro', 'african cup', 'gold cup',
        'asian cup', 'nations league', 'concacaf', 'conmebol',
        'uefa', 'afcon', 'coupe d\'afrique'
    ]):
        return 'copa_continental'
    elif 'friendly' in nombre:
        return 'amistoso'
    else:
        return 'otro'


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL DATASET INTERNACIONAL (2010+)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_dataset_internacional(conexion):
    """
    Lee el CSV 'selecciones-partidos-completos.csv', aplica limpieza y
    normalización, filtra desde 2010, y lo carga en la tabla
    'partidos_internacionales' de SQLite.

    Pipeline de limpieza aplicado:
    1. Filtrar desde el año 2010
    2. Eliminar filas sin resultado (NaN en home_score o away_score)
    3. Convertir scores a entero
    4. Normalizar nombres de equipos
    5. Clasificar torneos por categoría
    6. Eliminar duplicados
    """
    ruta = os.path.join(RAW_DATA_DIR, CSV_INTERNACIONAL)
    if not os.path.exists(ruta):
        print(f"  [AVISO] No se encontró {CSV_INTERNACIONAL}. Saltando.")
        return

    df = pd.read_csv(ruta)

    # ── Convertir fecha y extraer año ──
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year

    # ── Filtrar desde 2010 (datos relevantes para el nivel actual) ──
    total_orig = len(df)
    df = df[df['year'] >= 2010].copy()
    print(f"  Filtrado 2010+: {total_orig} -> {len(df)} partidos")

    # ── Eliminar filas sin resultado ──
    antes = len(df)
    df.dropna(subset=['home_score', 'away_score'], inplace=True)
    print(f"  Eliminados {antes - len(df)} registros sin resultado (NaN)")

    # ── Convertir scores a entero ──
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # ── Normalizar nombres de selecciones ──
    df['home_team'] = df['home_team'].apply(normalizar_nombre)
    df['away_team'] = df['away_team'].apply(normalizar_nombre)

    # ── Clasificar torneos ──
    df['tipo_torneo'] = df['tournament'].apply(clasificar_torneo)

    # ── Eliminar columna 'city' (no aporta al modelo) ──
    df.drop(columns=['city'], errors='ignore', inplace=True)

    # ── Convertir neutral a entero (True=1, False=0) ──
    df['neutral'] = df['neutral'].astype(int)

    # ── Fecha como string para SQLite ──
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # ── Eliminar duplicados ──
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Guardar en SQLite ──
    df.to_sql("partidos_internacionales", conexion, if_exists="replace", index=False)
    print(f"  [OK] partidos_internacionales: {len(df)} registros cargados")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE EQUIPOS 2026
# ─────────────────────────────────────────────────────────────────────────────
def cargar_equipos_2026(conexion):
    """Carga el CSV de selecciones clasificadas al Mundial 2026."""
    ruta = os.path.join(RAW_DATA_DIR, CSV_TEAMS_2026)
    if not os.path.exists(ruta):
        print(f"  [AVISO] No se encontró {CSV_TEAMS_2026}.")
        return

    df = pd.read_csv(ruta)
    df.rename(columns={'group': 'team_group'}, inplace=True)
    df['team'] = df['team'].str.strip()

    # Rellenar NaN en fifa_rank con 100 (posición penalización)
    df['fifa_rank'] = df['fifa_rank'].fillna(100).astype(int)

    df.to_sql("equipos_2026", conexion, if_exists="replace", index=False)
    print(f"  [OK] equipos_2026: {len(df)} selecciones cargadas")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE FIXTURES 2026
# ─────────────────────────────────────────────────────────────────────────────
def cargar_fixtures_2026(conexion):
    """Carga el calendario oficial de partidos del Mundial 2026."""
    ruta = os.path.join(RAW_DATA_DIR, CSV_FIXTURES_2026)
    if not os.path.exists(ruta):
        print(f"  [AVISO] No se encontró {CSV_FIXTURES_2026}.")
        return

    df = pd.read_csv(ruta)
    df.rename(columns={'group': 'team_group'}, inplace=True)
    df['team1'] = df['team1'].str.strip()
    df['team2'] = df['team2'].str.strip()

    for col in ['team1_fifa_rank', 'team2_fifa_rank']:
        if col in df.columns:
            df[col] = df[col].fillna(100)

    df.to_sql("fixtures_2026", conexion, if_exists="replace", index=False)
    print(f"  [OK] fixtures_2026: {len(df)} partidos del calendario cargados")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO DE INICIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def crear_tablas_y_cargar_datos():
    """
    Ejecuta el pipeline completo:
    1. Crea todas las tablas en SQLite
    2. Carga el dataset internacional (2010+)
    3. Carga selecciones y calendario del Mundial 2026
    """
    print("=" * 55)
    print("  INICIALIZANDO BASE DE DATOS")
    print("=" * 55)

    inicializar_base_de_datos()

    conexion = obtener_conexion()
    cargar_dataset_internacional(conexion)
    cargar_equipos_2026(conexion)
    cargar_fixtures_2026(conexion)
    conexion.close()

    print("\n[OK] Inicialización completada.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    crear_tablas_y_cargar_datos()
