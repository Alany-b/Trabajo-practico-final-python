import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import textwrap

def clean_html(html_str):
    """Elimina los espacios en blanco al inicio de cada línea para evitar que Streamlit renderice el HTML como un bloque de código Markdown."""
    return "\n".join(line.strip() for line in html_str.split("\n"))
# Intentamos importar la lógica local como fallback si la API de FastAPI no está activa
try:
    from src.predictor import predecir_partido_completo, ejecutar_monte_carlo, SimuladorMundial
    from src.elo import obtener_elo_equipo
    from src.utils import obtener_conexion, clasificar_torneo
    LIGAS_LOCALES_DISPONIBLES = True
except ImportError:
    LIGAS_LOCALES_DISPONIBLES = False

# Configuración de la URL base de la API FastAPI
API_URL = "http://127.0.0.1:8000"

# Ajustes de diseño de la página de Streamlit
st.set_page_config(
    page_title="World Cup 2026 AI Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado Premium (Glassmorphism, gradientes, minimalista)
st.markdown(clean_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;700;900&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #030712;
        color: #94a3b8;
    }

    .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 24px;
        font-size: 3.2rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    
    /* Glassmorphism Containers using data-element-id starts with glass_card */
    div[data-element-id^="glass_card"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(59, 130, 246, 0.15) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 24px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    
    div[data-element-id^="glass_card"]:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45) !important;
    }
    
    /* Metrics Styling */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, rgba(15,23,42,0.8) 0%, rgba(3,7,18,0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 20px;
    }
    
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 8px;
        color: #3b82f6;
    }

    /* Streamlit Overrides */
    div[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: #cbd5e1;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.45);
        color: #cbd5e1;
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Tabs styling (modernized, no red background) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    </style>
"""), unsafe_allow_html=True)


def test_api_conexion():
    """Comprueba si el servidor de la API FastAPI está activo."""
    try:
        response = requests.get(f"{API_URL}/")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Verificamos si podemos comunicarnos con el backend
api_activa = test_api_conexion()


# --- BARRA LATERAL (NAVEGACIÓN) ---
st.sidebar.markdown(clean_html("<h2 style='font-family: Outfit; font-weight: 800; color: #cbd5e1; margin-bottom: 0;'>AI Predictor</h2>"), unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #38bdf8; font-weight: 600; margin-bottom: 30px;'>World Cup 2026</p>", unsafe_allow_html=True)

opcion_menu = st.sidebar.radio(
    "",
    ["📊 Dashboard de Equipos", "⚽ Simulador 1 vs 1", "🏆 Simular Torneo", "🤖 Predictor Scikit-Learn"]
)

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
if api_activa:
    st.sidebar.markdown(clean_html("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 10px; border-radius: 8px; display: flex; align-items: center; gap: 8px;'>
            <div style='width: 8px; height: 8px; border-radius: 50%; background-color: #10b981; box-shadow: 0 0 8px #10b981;'></div>
            <span style='color: #10b981; font-size: 0.85rem; font-weight: 600;'>API Online</span>
        </div>
    """), unsafe_allow_html=True)
else:
    st.sidebar.markdown(clean_html("""
        <div style='background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); padding: 10px; border-radius: 8px; display: flex; align-items: center; gap: 8px;'>
            <div style='width: 8px; height: 8px; border-radius: 50%; background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b;'></div>
            <span style='color: #f59e0b; font-size: 0.85rem; font-weight: 600;'>Modo Local</span>
        </div>
    """), unsafe_allow_html=True)


# --- SECCIÓN 1: DASHBOARD DE EQUIPOS ---
if opcion_menu == "📊 Dashboard de Equipos":
    st.markdown("<h1 class='title-text'>Selecciones Clasificadas</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Analiza el estado actual, el ranking ELO y las métricas de las 48 selecciones que participarán en el Mundial 2026.</p>", unsafe_allow_html=True)

    # Obtenemos los datos de los equipos (Vía API o localmente)
    equipos_datos = []
    if api_activa:
        try:
            response = requests.get(f"{API_URL}/teams")
            if response.status_code == 200:
                equipos_datos = response.json()
        except Exception:
            pass

    if not equipos_datos and LIGAS_LOCALES_DISPONIBLES:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT e.team, e.team_group, e.confederation, e.fifa_rank, e.coach, r.rating
            FROM equipos_2026 e
            LEFT JOIN ratings_elo r ON e.team = r.team
            ORDER BY r.rating DESC
        """)
        equipos = cursor.fetchall()
        conexion.close()
        equipos_datos = [
            {
                "team": eq['team'],
                "team_group": eq['team_group'],
                "confederation": eq['confederation'],
                "fifa_rank": eq['fifa_rank'],
                "elo_rating": eq['rating'] if eq['rating'] is not None else 1500.0,
                "coach": eq['coach']
            } for eq in equipos
        ]

    if equipos_datos:
        df_equipos = pd.DataFrame(equipos_datos)
        
        # Panel superior de métricas
        col_m1, col_m2, col_m3 = st.columns(3)
        
        max_elo_team = df_equipos.loc[df_equipos['elo_rating'].idxmax(), 'team']
        max_elo_val = df_equipos['elo_rating'].max()
        best_fifa_team = df_equipos.loc[df_equipos['fifa_rank'].idxmin(), 'team']
        best_fifa_val = df_equipos['fifa_rank'].min()
        avg_elo_val = df_equipos['elo_rating'].mean()
        
        with col_m1:
            st.markdown(clean_html(f"""
                <div class="metric-container">
                    <div class="metric-title">Mejor Rating ELO</div>
                    <div class="metric-value" style="color: #fbbf24;">{max_elo_val:.0f}</div>
                    <div style="color: #cbd5e1; font-weight: 500; margin-top: 4px;">{max_elo_team}</div>
                </div>
            """), unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(clean_html(f"""
                <div class="metric-container">
                    <div class="metric-title">Mejor Rank FIFA</div>
                    <div class="metric-value" style="color: #34d399;">#{best_fifa_val}</div>
                    <div style="color: #cbd5e1; font-weight: 500; margin-top: 4px;">{best_fifa_team}</div>
                </div>
            """), unsafe_allow_html=True)
            
        with col_m3:
            st.markdown(clean_html(f"""
                <div class="metric-container">
                    <div class="metric-title">Promedio Global</div>
                    <div class="metric-value">{avg_elo_val:.0f}</div>
                    <div style="color: #cbd5e1; font-weight: 500; margin-top: 4px;">Puntos ELO</div>
                </div>
            """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Controles de filtro y tabla en una Card
        with st.container(key="glass_card_dashboard"):
            df_mostrar = df_equipos.rename(columns={
                "team": "Selección",
                "team_group": "Grupo",
                "confederation": "Confederación",
                "fifa_rank": "Ranking FIFA",
                "elo_rating": "Rating ELO",
                "coach": "Director Técnico"
            })
    
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filtro_conf = st.selectbox("🌍 Filtrar por Confederación:", ["Todas"] + list(df_mostrar["Confederación"].unique()))
            with col_f2:
                ordenar_por = st.selectbox("📋 Ordenar por:", ["Rating ELO (Descendente)", "Ranking FIFA (Ascendente)", "Grupo"])
    
            if filtro_conf != "Todas":
                df_mostrar = df_mostrar[df_mostrar["Confederación"] == filtro_conf]
    
            if "Rating ELO" in ordenar_por:
                df_mostrar = df_mostrar.sort_values(by="Rating ELO", ascending=False)
            elif "Ranking FIFA" in ordenar_por:
                df_mostrar = df_mostrar.sort_values(by="Ranking FIFA", ascending=True)
            else:
                df_mostrar = df_mostrar.sort_values(by="Grupo")
    
            # Configuración visual de la tabla (redondear ELO)
            df_mostrar["Rating ELO"] = df_mostrar["Rating ELO"].round(1)
            st.dataframe(df_mostrar.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error("No se pudieron cargar los datos. Inicializa la base de datos primero ejecutando python src/utils.py")


# --- SECCIÓN 2: SIMULADOR 1 VS 1 ---
elif opcion_menu == "⚽ Simulador 1 vs 1":
    st.markdown("<h1 class='title-text'>Simulador Cara a Cara</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Enfrenta a dos selecciones utilizando un modelo basado en Distribución de Poisson ajustado por Ratings ELO dinámicos.</p>", unsafe_allow_html=True)

    lista_equipos = []
    if LIGAS_LOCALES_DISPONIBLES:
        conexion = obtener_conexion()
        cursor = conexion.cursor()
        cursor.execute("SELECT team FROM equipos_2026 ORDER BY team ASC")
        lista_equipos = [row['team'] for row in cursor.fetchall()]
        conexion.close()

    if not lista_equipos and api_activa:
        try:
            response = requests.get(f"{API_URL}/teams")
            if response.status_code == 200:
                lista_equipos = [eq['team'] for eq in response.json()]
        except Exception:
            pass

    if lista_equipos:
        with st.container(key="glass_card_simulator"):
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                st.markdown(clean_html("<h3 style='color: #cbd5e1; font-family: Outfit; margin-bottom: 15px;'>Equipo Local</h3>"), unsafe_allow_html=True)
                eq_local = st.selectbox("Selección A:", lista_equipos, index=lista_equipos.index("Argentina") if "Argentina" in lista_equipos else 0, key="eq_a")
                
            with col_eq2:
                st.markdown(clean_html("<h3 style='color: #cbd5e1; font-family: Outfit; margin-bottom: 15px;'>Equipo Visitante</h3>"), unsafe_allow_html=True)
                eq_visitante = st.selectbox("Selección B:", lista_equipos, index=lista_equipos.index("France") if "France" in lista_equipos else 1, key="eq_b")
    
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 25px 0;'>", unsafe_allow_html=True)
            
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                es_neutral = st.checkbox("🌍 Jugar en cancha neutral (Mundial)", value=True)
            with col_opt2:
                usar_elo = st.checkbox("📈 Ajustar predicción usando Ranking ELO", value=True)
    
            st.markdown("<br>", unsafe_allow_html=True)
            btn_simular = st.button("🚀 Calcular Probabilidades Exactas")

        if btn_simular:
            if eq_local == eq_visitante:
                st.error("⚠️ Por favor, selecciona dos selecciones diferentes.")
            else:
                with st.spinner("Analizando historial y calculando matrices de probabilidad..."):
                    pred = None
                    if api_activa:
                        try:
                            payload = {
                                "team_local": eq_local,
                                "team_visitor": eq_visitante,
                                "is_neutral": es_neutral,
                                "use_elo": usar_elo
                            }
                            response = requests.post(f"{API_URL}/predict", json=payload)
                            if response.status_code == 200:
                                api_res = response.json()
                                pred = {
                                    'equipo_A': api_res['team_local'],
                                    'equipo_B': api_res['team_visitor'],
                                    'xG_A': api_res['xg_local'],
                                    'xG_B': api_res['xg_visitor'],
                                    'prob_A': api_res['prob_local'],
                                    'prob_empate': api_res['prob_draw'],
                                    'prob_B': api_res['prob_visitor'],
                                    'marcador_probable': api_res['most_probable_score'],
                                    'prob_marcador': api_res['prob_score']
                                }
                        except Exception:
                            pass

                    if pred is None and LIGAS_LOCALES_DISPONIBLES:
                        pred = predecir_partido_completo(eq_local, eq_visitante, es_neutral, usar_elo)

                    if pred:
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        prob_A_pct = pred['prob_A'] * 100
                        prob_draw_pct = pred['prob_empate'] * 100
                        prob_B_pct = pred['prob_B'] * 100

                        st.markdown(clean_html(f"""
                            <div style="background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(59, 130, 246, 0.15); border-radius: 16px; padding: 24px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);">
                                <h3 style="text-align: center; font-family: Outfit; color: #94a3b8; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 20px;">Resultado del Análisis</h3>
                                
                                <!-- Barra de progreso 1X2 -->
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-family: 'Outfit'; font-weight: 700;">
                                    <span style="color: #60a5fa;">{pred['equipo_A']} ({prob_A_pct:.1f}%)</span>
                                    <span style="color: #cbd5e1;">Empate ({prob_draw_pct:.1f}%)</span>
                                    <span style="color: #fbbf24;">{pred['equipo_B']} ({prob_B_pct:.1f}%)</span>
                                </div>
                                <div style="display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin-bottom: 40px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
                                    <div style="width: {prob_A_pct}%; background: #3b82f6;"></div>
                                    <div style="width: {prob_draw_pct}%; background: #475569;"></div>
                                    <div style="width: {prob_B_pct}%; background: #f59e0b;"></div>
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; text-align: center;">
                                    <div style="background: rgba(15,23,42,0.5); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                                        <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; margin-bottom: 5px;">Goles Esperados (xG)</div>
                                        <div style="color: #60a5fa; font-size: 2rem; font-weight: 800; font-family: Outfit;">{pred['xG_A']:.2f}</div>
                                        <div style="color: #cbd5e1; font-weight: 500;">{pred['equipo_A']}</div>
                                    </div>
                                    
                                    <div style="background: rgba(15,23,42,0.8); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 0 20px rgba(0,0,0,0.3);">
                                        <div style="color: #cbd5e1; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; margin-bottom: 5px;">Marcador Exacto Más Probable</div>
                                        <div style="color: #cbd5e1; font-size: 2.5rem; font-weight: 900; font-family: Outfit; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">{pred['marcador_probable'][0]} - {pred['marcador_probable'][1]}</div>
                                        <div style="color: #10b981; font-weight: 600; font-size: 0.9rem;">Confianza: {pred['prob_marcador']*100:.1f}%</div>
                                    </div>
                                    
                                    <div style="background: rgba(15,23,42,0.5); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                                        <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; margin-bottom: 5px;">Goles Esperados (xG)</div>
                                        <div style="color: #fbbf24; font-size: 2rem; font-weight: 800; font-family: Outfit;">{pred['xG_B']:.2f}</div>
                                        <div style="color: #cbd5e1; font-weight: 500;">{pred['equipo_B']}</div>
                                    </div>
                                </div>
                            </div>
                        """), unsafe_allow_html=True)
                    else:
                        st.error("No se pudo realizar la predicción. Revisa los logs.")


# --- SECCIÓN 3: SIMULAR TORNEO ---
elif opcion_menu == "🏆 Simular Torneo":
    st.markdown("<h1 class='title-text'>Simulador Monte Carlo</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Ejecuta miles de simulaciones del Mundial 2026 completo (Fase de Grupos + Llaves Eliminatorias) para encontrar a las selecciones matemáticamente más probables de alzar la copa.</p>", unsafe_allow_html=True)

    with st.container(key="glass_card_tournament_control"):
        iteraciones = st.slider("🔢 Número de universos paralelos a simular:", min_value=10, max_value=1000, value=100, step=10)
        st.markdown("<p style='font-size: 0.85rem; color: #64748b; margin-top: -10px; margin-bottom: 20px;'>Valores más altos entregan mayor precisión estadística pero toman más tiempo de procesamiento.</p>", unsafe_allow_html=True)
        
        btn_montecarlo = st.button("🌐 Iniciar Simulación Cuántica")

    if btn_montecarlo:
        with st.spinner(f"Simulando el Mundial 2026 {iteraciones} veces. Calculando probabilidades..."):
            df_mc = None
            
            if api_activa:
                try:
                    response = requests.get(f"{API_URL}/simulate-montecarlo?iteraciones={iteraciones}")
                    if response.status_code == 200:
                        df_mc = pd.DataFrame(response.json())
                except Exception:
                    pass

            if df_mc is None and LIGAS_LOCALES_DISPONIBLES:
                df_mc = ejecutar_monte_carlo(iteraciones=iteraciones)

            if df_mc is not None and not df_mc.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(clean_html("<h2 style='font-family: Outfit; color: #cbd5e1;'>🏆 Top 10 Candidatos al Título</h2>"), unsafe_allow_html=True)
                
                top_10 = df_mc.head(10).copy()
                top_10['Campeon_%'] = top_10['Campeon_%'].astype(float)
                
                import altair as alt
                
                # Gráfico premium con degradados
                chart = alt.Chart(top_10).mark_bar(
                    cornerRadiusTopRight=8,
                    cornerRadiusBottomRight=8,
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#38bdf8', offset=0),
                               alt.GradientStop(color='#818cf8', offset=1)],
                        x1=0, x2=1, y1=0, y2=0
                    )
                ).encode(
                    x=alt.X('Campeon_%:Q', title='Probabilidad de ser Campeón (%)', axis=alt.Axis(grid=True, gridColor='rgba(255,255,255,0.05)', labelColor='#94a3b8')),
                    y=alt.Y('Equipo:N', sort=alt.EncodingSortField(field='Campeon_%', order='descending'), title=''),
                    tooltip=[
                        alt.Tooltip('Equipo:N', title='Selección'),
                        alt.Tooltip('Campeon_%:Q', title='Probabilidad de Campeonar', format='.2f')
                    ]
                ).properties(
                    height=400
                ).configure_view(
                    strokeWidth=0,
                    fill='transparent'
                ).configure_axis(
                    labelFont='Inter',
                    titleFont='Outfit',
                    labelFontSize=12,
                    titleFontSize=14,
                    domainColor='transparent'
                )
                
                with st.container(key="glass_card_tournament_chart"):
                    st.altair_chart(chart, use_container_width=True)

                st.markdown(clean_html("<h3 style='font-family: Outfit; color: #cbd5e1; margin-top: 30px;'>📊 Matriz Completa de Probabilidades</h3>"), unsafe_allow_html=True)
                with st.container(key="glass_card_tournament_matrix"):
                    # Dar formato visual a las probabilidades
                    df_formateado = df_mc.copy()
                    columnas_pct = ['Clasifica_Grupo_%', 'Llega_Cuartos_%', 'Llega_Semis_%', 'Llega_Final_%', 'Campeon_%']
                    for col in columnas_pct:
                        df_formateado[col] = df_formateado[col].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(df_formateado, use_container_width=True, hide_index=True)
            else:
                st.error("No se pudo procesar la simulación. Asegúrate de haber ejecutado src/utils.py primero.")

# --- SECCIÓN 4: PREDICTOR MACHINE LEARNING (Requisito Académico) ---
elif opcion_menu == "🤖 Predictor Scikit-Learn":
    st.markdown("<h1 class='title-text'>Predictor Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Evaluación del modelo Random Forest entrenado en Scikit-Learn y exportado vía joblib.</p>", unsafe_allow_html=True)

    lista_equipos = []
    if LIGAS_LOCALES_DISPONIBLES:
        from src.utils import obtener_conexion
        try:
            conexion = obtener_conexion()
            cursor = conexion.cursor()
            cursor.execute("SELECT team FROM equipos_2026 ORDER BY team ASC")
            lista_equipos = [row[0] for row in cursor.fetchall()]
            conexion.close()
        except Exception:
            lista_equipos = ["Argentina", "France", "Brazil", "England", "Spain"]
    else:
        lista_equipos = ["Argentina", "France", "Brazil", "England", "Spain"]

    with st.container(key="glass_card_ml"):
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.markdown(clean_html("<h3 style='color: #cbd5e1; font-family: Outfit; margin-bottom: 15px;'>Equipo Local</h3>"), unsafe_allow_html=True)
            eq_local = st.selectbox("Selección Local:", lista_equipos, index=lista_equipos.index("Argentina") if "Argentina" in lista_equipos else 0, key="ml_eq_a")
            
        with col_eq2:
            st.markdown(clean_html("<h3 style='color: #cbd5e1; font-family: Outfit; margin-bottom: 15px;'>Equipo Visitante</h3>"), unsafe_allow_html=True)
            eq_visitante = st.selectbox("Selección Visitante:", lista_equipos, index=lista_equipos.index("France") if "France" in lista_equipos else 1, key="ml_eq_b")
    
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 25px 0;'>", unsafe_allow_html=True)
        es_neutral = st.checkbox("⚽ Jugar en cancha neutral (Mundial)", value=True, key="ml_neutral")
    
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 Ejecutar Modelo Random Forest"):
            if eq_local == eq_visitante:
                st.warning("Por favor, selecciona dos equipos diferentes.")
            else:
                with st.spinner("Consultando al modelo Scikit-Learn..."):
                    payload = {
                        "team_local": eq_local,
                        "team_visitor": eq_visitante,
                        "is_neutral": es_neutral,
                        "use_elo": True
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/predict-ml", json=payload)
                        if response.status_code == 200:
                            pred_ml = response.json()
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            prob_l_pct = pred_ml['prob_local'] * 100
                            prob_e_pct = pred_ml['prob_draw'] * 100
                            prob_v_pct = pred_ml['prob_visitor'] * 100
                            
                            st.markdown(clean_html(f"""
                                <div style="background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(59, 130, 246, 0.15); border-radius: 16px; padding: 24px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);">
                                    <h3 style="text-align: center; font-family: Outfit; color: #94a3b8; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 20px;">Predicción Scikit-Learn</h3>
                                    
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-family: 'Outfit'; font-weight: 700;">
                                        <span style="color: #60a5fa;">{pred_ml['team_local']} ({prob_l_pct:.1f}%)</span>
                                        <span style="color: #cbd5e1;">Empate ({prob_e_pct:.1f}%)</span>
                                        <span style="color: #fbbf24;">{pred_ml['team_visitor']} ({prob_v_pct:.1f}%)</span>
                                    </div>
                                    <div style="display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin-bottom: 40px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
                                        <div style="width: {prob_l_pct}%; background: #3b82f6;"></div>
                                        <div style="width: {prob_e_pct}%; background: #475569;"></div>
                                        <div style="width: {prob_v_pct}%; background: #f59e0b;"></div>
                                    </div>
                                    
                                    <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.1);">
                                        <p style="color: #cbd5e1; text-transform: uppercase; font-size: 0.85rem; font-weight: 600; margin-bottom: 5px;">Resultado Más Probable</p>
                                        <h2 style="color: #cbd5e1; font-family: Outfit; font-weight: 900; margin: 0;">{pred_ml['predicted_result']}</h2>
                                    </div>
                                </div>
                            """), unsafe_allow_html=True)
                        else:
                            st.error(f"Error de la API: {response.text}")
                    except Exception as e:
                        st.error(f"Error al conectar con la API: {e}")
