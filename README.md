# ⚽ Predicción del Mundial 2026 — Solución IA

Este proyecto representa una solución académica integral basada en **Machine Learning** que abarca todo el ciclo de vida de un proyecto de datos: desde la adquisición, exploración y limpieza de datos, hasta el entrenamiento de clasificadores y el despliegue de una interfaz interactiva y una API web.

---

## 📋 Cumplimiento de los Requerimientos

Aquí se detalla cómo el proyecto responde a cada consigna de la materia:

### 1. Datos y Dataset
* **Origen:** Resultados históricos de partidos internacionales de fútbol masculino (1872 - 2026).
* **Preprocesamiento:** Filtrado para conservar solo partidos del fútbol moderno (año 2010 en adelante), limpieza de registros sin goles y normalización de nombres de países para coincidir con el fixture oficial del Mundial 2026.

### 2. Notebooks de Ciencia de Datos
El trabajo práctico está dividido de forma ordenada en la carpeta `notebooks/`:
* [**`03_limpieza_dataset_internacional.ipynb`**](file:///C:/Users/IPF-2026/Desktop/prediccion-mundial/notebooks/03_limpieza_dataset_internacional.ipynb): Contiene la exploración inicial (EDA), gráficos de distribución de datos, mapa de calor de valores nulos y exportación de datos limpios.
* [**`04_entrenamiento_scikit.ipynb`**](file:///C:/Users/IPF-2026/Desktop/prediccion-mundial/notebooks/04_entrenamiento_scikit.ipynb): Contiene la ingeniería de características (ataque y defensa por selección), división temporal train/test, entrenamiento de modelos (Random Forest, Regresión Logística, Gradient Boosting), comparación de métricas y la exportación final del modelo usando `joblib`.

### 3. Servidor de Predicción (Backend)
* Desarrollado en **FastAPI** (`app/main.py`).
* Carga automáticamente el modelo serializado `modelo_rf.pkl` y las estadísticas al arrancar.
* Expone un endpoint REST (`POST /predict-ml`) para recibir consultas y devolver probabilidades del partido.

### 4. Interfaz de Usuario (Frontend)
* Desarrollada en **Streamlit** (`app/frontend.py`).
* Dashboard dinámico y estético que consume la API REST, con una sección específica llamada **Predictor Scikit-Learn** donde se ingresan equipos para visualizar porcentajes de Victoria, Empate y Derrota.

---

## 📁 Estructura del Proyecto

La estructura de carpetas se mantiene organizada de la siguiente manera:

```text
prediccion-mundial/
│
├── datasets/                     # Almacenamiento de archivos de datos
│   ├── raw/                      # Dataset original sin procesar (.csv)
│   └── processed/                # Dataset limpio (.csv) y base de datos SQLite (.db)
│
├── notebooks/                    # Jupyter Notebooks requeridos para la materia
│   ├── 03_limpieza_dataset.ipynb # Pipeline de limpieza y análisis exploratorio (EDA)
│   └── 04_entrenamiento.ipynb    # Entrenamiento de algoritmos y exportación del modelo
│
├── src/                          # Módulos Python con la lógica de negocio secundaria
│   ├── utils.py                  # Utilidades y conexiones a base de datos
│   ├── elo.py                    # Algoritmo de cálculo de ratings ELO
│   └── poisson.py                # Modelo probabilístico de Poisson
│
├── app/                          # Código de la aplicación web
│   ├── main.py                   # API Backend (FastAPI)
│   ├── schemas.py                # Esquemas y validaciones de datos (Pydantic)
│   └── frontend.py               # Servidor Frontend (Streamlit)
│
├── models/                       # Binarios exportados del modelo de Machine Learning
│   ├── modelo_rf.pkl             # Modelo Random Forest serializado (joblib)
│   └── estadisticas.pkl          # Diccionario con ratings de ataque y defensa calculados
│
├── requirements.txt              # Listado de dependencias del entorno de desarrollo
└── README.md                     # Este documento explicativo
```

---

## ⚙️ Guía de Instalación y Configuración

Asegúrate de estar en la raíz de la carpeta `prediccion-mundial` antes de correr los comandos.

### 1. Preparar el Entorno Virtual (Venv)

Elige **una** de las siguientes opciones de configuración según tus herramientas instaladas:

#### Opción A: Configuración Tradicional de Python (Recomendado)
*Si deseas utilizar el instalador estándar `pip` que viene con Python:*

1. **Eliminar el entorno virtual anterior** (si existe uno roto):
   ```powershell
   Remove-Item -Recurse -Force .venv
   ```
2. **Crear un nuevo entorno virtual limpio**:
   ```powershell
   python -m venv .venv
   ```
3. **Activar el entorno virtual**:
   * **PowerShell (VS Code)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Símbolo del Sistema (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
4. **Instalar las dependencias de Python**:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

#### Opción B: Configuración Rápida con `uv`
*Si tu entorno inicial fue creado con la herramienta `uv` (que no incluye `pip` por defecto):*

1. **Instalar dependencias mediante `uv`**:
   ```powershell
   uv pip install -r requirements.txt
   ```
2. **Instalar uvicorn explícitamente en el entorno**:
   ```powershell
   uv pip install uvicorn[standard]
   ```

---

## 🚀 Guía de Ejecución del Proyecto

Una vez que el entorno virtual esté activo y las librerías instaladas, sigue estos pasos para levantar la aplicación:

### Paso 1: Iniciar el Servidor Backend (FastAPI)
Este servidor cargará el modelo pre-entrenado y procesará las predicciones. Corre el comando según la opción de entorno elegida:

* **Para la Opción A (Python Estándar)**:
  ```powershell
  python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
  ```
* **Para la Opción B (Herramienta `uv`)**:
  ```powershell
  uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
  ```

> 💡 *Puedes ver la documentación interactiva de la API y probar sus endpoints en: http://127.0.0.1:8000/docs*

---

### Paso 2: Iniciar la Interfaz de Usuario (Streamlit)
**Abre una segunda pestaña de terminal independiente**, vuelve a activar el entorno virtual (`.\.venv\Scripts\Activate.ps1`) y ejecuta la interfaz del frontend:

* **Para la Opción A (Python Estándar)**:
  ```powershell
  streamlit run app/frontend.py
  ```
* **Para la Opción B (Herramienta `uv`)**:
  ```powershell
  uv run streamlit run app/frontend.py
  ```

> 🌐 *Accede a la interfaz web abriendo en tu navegador la dirección: http://localhost:8501*  
> *Una vez allí, ve al menú lateral izquierdo y haz clic en **"🤖 Predictor Scikit-Learn"** para simular partidos con el modelo final entrenado.*
