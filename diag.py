import pandas as pd

df = pd.read_csv('datasets/raw/selecciones-partidos-completos.csv')
print(f"Shape: {df.shape}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nPrimeras filas:")
print(df.head(5).to_string())
print(f"\nTorneos (top 20):")
print(df['tournament'].value_counts().head(20).to_string())
print(f"\nNulos:")
print(df.isnull().sum().to_string())
print(f"\nRango de fechas:")
print(f"  Desde: {df['date'].min()}")
print(f"  Hasta: {df['date'].max()}")
print(f"\nPartidos de Argentina (últimos 10):")
arg = df[(df['home_team']=='Argentina') | (df['away_team']=='Argentina')]
print(arg.tail(10)[['date','home_team','home_score','away_score','away_team','tournament']].to_string())
print(f"\nPartidos de France (últimos 5):")
fra = df[(df['home_team']=='France') | (df['away_team']=='France')]
print(fra.tail(5)[['date','home_team','home_score','away_score','away_team','tournament']].to_string())
