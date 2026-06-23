import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print('Cargando datos...')
df = pd.read_csv('datasets/processed/partidos_internacionales_limpios.csv')

def determinar_resultado(row):
    if row['home_score'] > row['away_score']: return 2
    elif row['home_score'] == row['away_score']: return 1
    else: return 0

df['resultado'] = df.apply(determinar_resultado, axis=1)

df_home = df.groupby('home_team').agg(
    goles_favor_local=('home_score', 'mean'),
    goles_contra_local=('away_score', 'mean')
).reset_index().rename(columns={'home_team': 'equipo'})

df_away = df.groupby('away_team').agg(
    goles_favor_visit=('away_score', 'mean'),
    goles_contra_visit=('home_score', 'mean')
).reset_index().rename(columns={'away_team': 'equipo'})

estadisticas = pd.merge(df_home, df_away, on='equipo', how='outer').fillna(0)
estadisticas['ataque'] = (estadisticas['goles_favor_local'] + estadisticas['goles_favor_visit']) / 2
estadisticas['defensa'] = (estadisticas['goles_contra_local'] + estadisticas['goles_contra_visit']) / 2

df = df.merge(estadisticas[['equipo','ataque','defensa']], left_on='home_team', right_on='equipo', how='left')
df.rename(columns={'ataque': 'ataque_local', 'defensa': 'defensa_local'}, inplace=True)
df.drop(columns=['equipo'], inplace=True)

df = df.merge(estadisticas[['equipo','ataque','defensa']], left_on='away_team', right_on='equipo', how='left')
df.rename(columns={'ataque': 'ataque_visit', 'defensa': 'defensa_visit'}, inplace=True)
df.drop(columns=['equipo'], inplace=True)

df['es_neutral'] = df['neutral'].astype(int)

torneo_map = {'mundial': 4, 'eliminatoria': 3, 'copa_continental': 2, 'amistoso': 1, 'otro': 1}
df['peso_torneo'] = df['tipo_torneo'].map(torneo_map).fillna(1)

FEATURES = ['ataque_local', 'defensa_local', 'ataque_visit', 'defensa_visit', 'es_neutral', 'peso_torneo']
df_modelo = df[FEATURES + ['resultado', 'year']].dropna()

X = df_modelo[FEATURES]
y = df_modelo['resultado']
years = df_modelo['year']

X_train = X[years < 2023]
X_test  = X[years >= 2023]
y_train = y[years < 2023]
y_test  = y[years >= 2023]

print('Entrenando Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

acc = rf.score(X_test, y_test)
print(f'Test Accuracy: {acc:.3f}')

os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/modelo_rf.pkl')
joblib.dump(estadisticas, 'models/estadisticas.pkl')
print('Modelos guardados exitosamente.')
