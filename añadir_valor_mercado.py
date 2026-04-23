"""
añadir_valor_mercado.py
========================
Añade la columna valor_mercado_millones_eur al dataset_modelo.csv
usando el archivo valor_mercado_2026.csv.

Fuente de los datos:
  - Transfermarkt, diciembre 2025
  - Reportados por ESPN, sportsorca.com y beinsports.com
  - Referencia ESPN: https://www.espn.com/soccer/story/_/id/48361967/2026-world-cup-squads-ranked-all-48-national-teams-win-tournament

Uso:
  python añadir_valor_mercado.py
"""

import pandas as pd

print("💰 Añadiendo valor de mercado al dataset...")

# Cargar dataset principal
df = pd.read_csv('dataset_modelo.csv')

# Cargar valores de mercado
mercado = pd.read_csv('valor_mercado_2026.csv')[['equipo', 'valor_mercado_millones_eur']]

# Unir
df = df.merge(mercado, on='equipo', how='left')

# Para equipos sin valor (si hubiera alguno), usar la mediana
mediana = df['valor_mercado_millones_eur'].median()
df['valor_mercado_millones_eur'].fillna(mediana, inplace=True)

# Guardar
df.to_csv('dataset_modelo.csv', index=False)

print(f"✅ Valor de mercado añadido a {df['valor_mercado_millones_eur'].notna().sum()} equipos")
print(f"\nTop 10 equipos por valor de mercado:")
print(df[['equipo', 'valor_mercado_millones_eur']].sort_values(
    'valor_mercado_millones_eur', ascending=False).head(10).to_string(index=False))
