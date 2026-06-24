# ⚽ Trabajo Práctico Final: Solución de IA Integral — Mundial 2026

Este proyecto representa una solución integral basada en Machine Learning que cumple con todo el ciclo de vida de un proyecto de datos, desde la limpieza y exploración hasta el despliegue de una API (FastAPI) y una interfaz gráfica interactiva (Streamlit).

## ✅ Cumplimiento de los Requerimientos del Proyecto

### 1. Requerimientos del Dataset
- **Origen:** Dataset de Resultados de Fútbol Internacional (Público).
- **Registros:** +49,000 partidos procesados y filtrados a ~15,000 desde el año 2010.
- **Atributos:** Múltiples variables incluyendo fecha, equipo local, equipo visitante, torneo, neutralidad, goles, etc.

### 2. El Notebook de Ciencia de Datos
Se han separado las responsabilidades en dos notebooks principales debidamente documentados en Markdown:
- `notebooks/03_limpieza_dataset_internacional.ipynb`: Contiene el **Análisis Exploratorio de Datos (EDA)** con Pandas/Matplotlib/Seaborn y el **Preprocesamiento** (limpieza de nulos, estandarización).
- `notebooks/04_entrenamiento_scikit.ipynb`: Contiene el **Modelado** (entrenamiento y evaluación comparativa de Regresión Logística, Random Forest, etc.) y la **Exportación** del modelo final `.pkl` utilizando la librería `joblib`.

### 3. Implementación del Servidor (Backend)
- Desarrollado usando **FastAPI** (`app/main.py`).
- Importa `joblib` al inicializarse y carga el archivo `models/modelo_rf.pkl` pre-entrenado.
- Expone el endpoint **`POST /predict-ml`** que recibe un JSON con los equipos y devuelve la predicción probabilística calculada por el modelo Scikit-Learn.
- *Opcional Académico:* El servidor también aloja un modelo avanzado basado en Distribución de Poisson acoplado a un motor ELO dinámico.

### 4. Interfaz de Usuario (Frontend)
- Desarrollada utilizando **Streamlit** (`app/frontend.py`).
- Interfaz web amigable, altamente personalizada con CSS moderno (Glassmorphism), que consume la API REST.
- Posee una sección dedicada "🤖 Predictor Scikit-Learn" donde el usuario puede ingresar datos (local, visitante, neutralidad) y visualizar claramente el porcentaje de victoria, empate o derrota dictado por el modelo exportado.

### 5. Entregables y Estructura del Proyecto
- **Repositorio GitHub:** [Enlace a proveer por el estudiante]
- **README.md:** Este documento con las instrucciones.
- **Requirements.txt:** Contiene todas las dependencias (`joblib`, `scikit-learn`, `fastapi`, `streamlit`, etc.).
- **models/**: Contiene `modelo_rf.pkl` (modelo) y `estadisticas.pkl` (features).
- **Código Fuente:** Correctamente modularizado en `app/` y `src/`.

---

## 📁 Estructura de Archivos

```text
prediccion-mundial/
├── datasets/
│   ├── raw/
│   └── processed/
│       ├── futbol_predictivo.db
│       └── partidos_internacionales_limpios.csv
│
├── src/                          # Lógica del modelo (código Python)
│   ├── utils.py                  # Conexión SQLite + carga inicial
│   ├── elo.py                    # Rating ELO con K-factor dinámico
│   ├── poisson.py                # Distribución de Poisson
│   └── predictor.py              # Motor de predicción Monte Carlo
│
├── app/                          # Aplicación web (Backend & Frontend)
│   ├── main.py                   # API REST con FastAPI (Endpoint /predict-ml)
│   ├── schemas.py                # Esquemas Pydantic
│   └── frontend.py               # Dashboard con Streamlit
│
├── notebooks/                    # Jupyter Notebooks requeridos
│   ├── 03_limpieza_dataset_internacional.ipynb  # EDA + Preprocesamiento
│   └── 04_entrenamiento_scikit.ipynb            # ML + Exportación joblib
│
├── models/                       # Modelos entrenados (.pkl)
│   ├── modelo_rf.pkl             # Modelo Random Forest (Requisito)
│   ├── estadisticas.pkl          # Features del modelo
│   └── ...
│
├── requirements.txt              # Dependencias
└── README.md                     # Documentación
```

---

## ⚙️ Guía de Inicialización

Asegúrate de estar en el entorno virtual (`.venv`) activado en la raíz del proyecto, en caso de no estar
o no tenerlo, debe ser creado utilizando el comando:

windows: py -m venv "nombre" y para activar "nombre"/Script/activate
linux: python3 -m venv "nombre" y para activar source "nombre"/bin/Activate

### 1. Instalar Dependencias
```powershell
pip install -r requirements.txt
```

### 2. (Opcional) Re-entrenar y Exportar el Modelo
Si deseas volver a generar el modelo `.pkl`:
Ejecuta las celdas del archivo `notebooks/04_entrenamiento_scikit.ipynb` en Jupyter Notebook. Al final del notebook, la librería `joblib` sobreescribirá los archivos en la carpeta `models/`.

### 3. Iniciar la API Backend (FastAPI)
Ejecuta el servidor. FastAPI cargará automáticamente el archivo `.pkl` de la carpeta `models/`:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
*(Puedes ver la documentación interactiva en http://127.0.0.1:8000/docs)*

### 4. Iniciar la Interfaz Web (Streamlit)
En otra terminal (con la API ya en ejecución):
```powershell
streamlit run app/frontend.py
```
Abre tu navegador, dirígete a http://localhost:8501 y en el menú lateral selecciona **"🤖 Predictor Scikit-Learn"** para probar el modelo de Machine Learning evaluado en el trabajo.
