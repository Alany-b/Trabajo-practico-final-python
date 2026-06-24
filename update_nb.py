import json
import os

notebook_path = 'notebooks/04_entrenamiento_scikit.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell para importar joblib y exportar
md_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Exportación del Modelo (Requisito Académico)\n",
    "Usamos joblib para exportar el modelo entrenado y los datos de estadísticas para ser consumidos por la API FastAPI."
   ]
}

code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os\n",
    "\n",
    "# Asegurarnos de que el directorio models exista\n",
    "os.makedirs('../models', exist_ok=True)\n",
    "\n",
    "# Guardamos el modelo Random Forest\n",
    "joblib.dump(rf, '../models/modelo_rf.pkl')\n",
    "print('Modelo Random Forest exportado a models/modelo_rf.pkl')\n",
    "\n",
    "# Guardamos las estadísticas para generar features en la API\n",
    "joblib.dump(estadisticas, '../models/estadisticas.pkl')\n",
    "print('Estadísticas exportadas a models/estadisticas.pkl')"
   ]
}

nb['cells'].extend([md_cell, code_cell])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook updated.")
