"""
preparar_datos.py
=================
Lee todos los CSVs del repo y construye el dataset final con todas las variables
necesarias para el modelo predictivo del Mundial 2026.

Variables que construye:
  - racha_reciente      : % victorias en últimos 10 partidos antes del mundial
  - goles_favor_avg     : promedio de goles anotados (histórico)
  - goles_contra_avg    : promedio de goles recibidos (histórico)
  - diferencia_goles    : goles_favor_avg - goles_contra_avg
  - fue_a_penales_pct   : % de veces que fue a penales en mundiales
  - gano_penales_pct    : % victorias en tandas de penales
  - ranking_fifa        : ranking FIFA actual (del CSV ya descargado)
  - es_local            : 1 si el equipo es USA, Canadá o México (sede 2026)
  - confederacion_cod   : UEFA=1, CONMEBOL=2, CAF=3, AFC=4, CONCACAF=5, OFC=6
  - interes_google      : puntuación de Google Trends (0-100) últimos 6 meses
  - capacidad_estadio   : capacidad promedio de los estadios donde juega
  - gano                : variable objetivo (1=ganó el mundial, 0=no)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def read_csv_auto(path):
    """Lee CSVs con codificaciones comunes y soporta archivos Excel renombrados a .csv."""
    with open(path, "rb") as f:
        signature = f.read(2)
    if signature == b"PK":
        return pd.read_excel(path)

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
    return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

# ─────────────────────────────────────────────
# 1. CARGAR ARCHIVOS BASE
# ─────────────────────────────────────────────
print("📂 Cargando archivos CSV...")

results      = read_csv_auto('results.csv')
goalscorers  = read_csv_auto('goalscorers.csv')
shootouts    = read_csv_auto('shootouts.csv')
ranking_fifa = read_csv_auto('ranking_fifa_masculino.csv')
stadiums     = read_csv_auto('stadiums.csv')
equipos      = read_csv_auto('equipos_mundial.csv')

print(f"  results.csv       : {len(results)} filas")
print(f"  goalscorers.csv   : {len(goalscorers)} filas")
print(f"  shootouts.csv     : {len(shootouts)} filas")
print(f"  ranking_fifa.csv  : {len(ranking_fifa)} filas")
print(f"  equipos.csv       : {len(equipos)} filas")

# ─────────────────────────────────────────────
# 2. FILTRAR SOLO PARTIDOS DE MUNDIALES FIFA
# ─────────────────────────────────────────────
# Ajusta el nombre de la columna si es diferente en tu results.csv
# Columnas típicas: date, home_team, away_team, home_score, away_score, tournament, country, city, neutral

print("\n🔍 Columnas en results.csv:", results.columns.tolist())

# Filtrar mundiales (ajusta el valor exacto si es diferente)
mundiales = results[results['tournament'].str.contains('FIFA World Cup', na=False, case=False)].copy()
mundiales['date'] = pd.to_datetime(mundiales['date'])
print(f"\n⚽ Partidos de Mundial encontrados: {len(mundiales)}")

# ─────────────────────────────────────────────
# 3. EXTRAER TODOS LOS EQUIPOS QUE PARTICIPARON
# ─────────────────────────────────────────────
equipos_local  = mundiales[['home_team']].rename(columns={'home_team': 'equipo'})
equipos_visita = mundiales[['away_team']].rename(columns={'away_team': 'equipo'})
todos_equipos  = pd.concat([equipos_local, equipos_visita]).drop_duplicates()
todos_equipos  = todos_equipos.reset_index(drop=True)
print(f"📋 Equipos únicos en mundiales históricos: {len(todos_equipos)}")

# ─────────────────────────────────────────────
# 4. VARIABLE: RACHA RECIENTE (últimos 10 partidos)
# ─────────────────────────────────────────────
print("\n📊 Calculando racha reciente...")

def calcular_racha(equipo, df_resultados, n=10):
    """% de victorias en los últimos n partidos de un equipo"""
    partidos_equipo = df_resultados[
        (df_resultados['home_team'] == equipo) |
        (df_resultados['away_team'] == equipo)
    ].sort_values('date').tail(n)

    victorias = 0
    for _, p in partidos_equipo.iterrows():
        if p['home_team'] == equipo and p['home_score'] > p['away_score']:
            victorias += 1
        elif p['away_team'] == equipo and p['away_score'] > p['home_score']:
            victorias += 1

    return victorias / len(partidos_equipo) if len(partidos_equipo) > 0 else 0

todos_equipos['racha_reciente'] = todos_equipos['equipo'].apply(
    lambda e: calcular_racha(e, results)  # usa results completo (no solo mundiales)
)

# ─────────────────────────────────────────────
# 5. VARIABLE: GOLES PROMEDIO (histórico en mundiales)
# ─────────────────────────────────────────────
print("📊 Calculando promedios de goles...")

def calcular_goles_promedio(equipo, df_mundiales):
    """Promedio de goles anotados y recibidos en mundiales"""
    como_local  = df_mundiales[df_mundiales['home_team'] == equipo]
    como_visita = df_mundiales[df_mundiales['away_team'] == equipo]

    goles_favor  = list(como_local['home_score']) + list(como_visita['away_score'])
    goles_contra = list(como_local['away_score']) + list(como_visita['home_score'])

    gf_avg = np.mean(goles_favor)  if goles_favor  else 0
    gc_avg = np.mean(goles_contra) if goles_contra else 0
    return round(gf_avg, 3), round(gc_avg, 3)

goles = todos_equipos['equipo'].apply(lambda e: calcular_goles_promedio(e, mundiales))
todos_equipos['goles_favor_avg']  = goles.apply(lambda x: x[0])
todos_equipos['goles_contra_avg'] = goles.apply(lambda x: x[1])
todos_equipos['diferencia_goles'] = todos_equipos['goles_favor_avg'] - todos_equipos['goles_contra_avg']

# ─────────────────────────────────────────────
# 6. VARIABLE: PENALES
# ─────────────────────────────────────────────
print("📊 Calculando estadísticas de penales...")

print("  Columnas shootouts.csv:", shootouts.columns.tolist())

# Ajusta según tus columnas reales
# Columnas típicas: date, home_team, away_team, winner

def calcular_penales(equipo, df_shootouts):
    participaciones = df_shootouts[
        (df_shootouts['home_team'] == equipo) |
        (df_shootouts['away_team'] == equipo)
    ]
    if len(participaciones) == 0:
        return 0, 0
    victorias_penales = len(participaciones[participaciones['winner'] == equipo])
    return len(participaciones), victorias_penales

penales = todos_equipos['equipo'].apply(lambda e: calcular_penales(e, shootouts))
todos_equipos['veces_penales']   = penales.apply(lambda x: x[0])
todos_equipos['victorias_penales'] = penales.apply(lambda x: x[1])
todos_equipos['gano_penales_pct'] = np.where(
    todos_equipos['veces_penales'] > 0,
    todos_equipos['victorias_penales'] / todos_equipos['veces_penales'],
    0.5  # si nunca fue a penales, asignamos 50% (neutral)
)

# ─────────────────────────────────────────────
# 7. VARIABLE: RANKING FIFA
# ─────────────────────────────────────────────
print("📊 Añadiendo ranking FIFA...")
print("  Columnas ranking_fifa.csv:", ranking_fifa.columns.tolist())

# Soporta schema antiguo (country_full/rank_date) y actual (pais/fecha_publicacion).
team_col = 'country_full' if 'country_full' in ranking_fifa.columns else 'pais'
date_col = 'rank_date' if 'rank_date' in ranking_fifa.columns else 'fecha_publicacion'
rank_col = 'rank' if 'rank' in ranking_fifa.columns else 'posicion'

if date_col in ranking_fifa.columns:
    ranking_fifa[date_col] = pd.to_datetime(ranking_fifa[date_col], errors='coerce')
    ranking_reciente = (
        ranking_fifa.sort_values(date_col, ascending=False)
        .groupby(team_col, as_index=False)
        .first()
    )
else:
    ranking_reciente = ranking_fifa.drop_duplicates(subset=[team_col])

ranking_reciente = ranking_reciente[[team_col, rank_col]].rename(
    columns={team_col: 'equipo', rank_col: 'ranking_fifa'}
)

todos_equipos = todos_equipos.merge(ranking_reciente, on='equipo', how='left')
fallback_rank = (todos_equipos['ranking_fifa'].max() + 10) if todos_equipos['ranking_fifa'].notna().any() else 999
todos_equipos['ranking_fifa'].fillna(fallback_rank, inplace=True)

# ─────────────────────────────────────────────
# 8. VARIABLE: ES LOCAL (sede del Mundial 2026)
# ─────────────────────────────────────────────
print("📊 Añadiendo variable sede 2026...")

sedes_2026 = ['United States', 'Canada', 'Mexico', 'USA']
todos_equipos['es_local'] = todos_equipos['equipo'].apply(
    lambda e: 1 if any(sede.lower() in e.lower() for sede in sedes_2026) else 0
)

# ─────────────────────────────────────────────
# 9. VARIABLE: CONFEDERACIÓN
# ─────────────────────────────────────────────
print("📊 Añadiendo confederaciones...")

confederaciones = {
    # UEFA (Europa)
    'Germany': 'UEFA', 'France': 'UEFA', 'Spain': 'UEFA', 'Italy': 'UEFA',
    'England': 'UEFA', 'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Belgium': 'UEFA',
    'Croatia': 'UEFA', 'Switzerland': 'UEFA', 'Denmark': 'UEFA', 'Serbia': 'UEFA',
    'Poland': 'UEFA', 'Austria': 'UEFA', 'Turkey': 'UEFA', 'Czech Republic': 'UEFA',
    'Hungary': 'UEFA', 'Romania': 'UEFA', 'Slovakia': 'UEFA', 'Slovenia': 'UEFA',
    'Scotland': 'UEFA', 'Wales': 'UEFA', 'Ukraine': 'UEFA', 'Greece': 'UEFA',

    # CONMEBOL (Sudamérica)
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL',
    'Ecuador': 'CONMEBOL', 'Paraguay': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',

    # CONCACAF (Norte/Centro América y Caribe)
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Costa Rica': 'CONCACAF', 'Honduras': 'CONCACAF', 'Jamaica': 'CONCACAF',
    'Panama': 'CONCACAF', 'El Salvador': 'CONCACAF', 'Haiti': 'CONCACAF',
    'Trinidad and Tobago': 'CONCACAF',

    # CAF (África)
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Ghana': 'CAF', 'Nigeria': 'CAF',
    'Cameroon': 'CAF', 'Tunisia': 'CAF', 'Egypt': 'CAF', 'Algeria': 'CAF',
    'Ivory Coast': 'CAF', 'Mali': 'CAF', 'South Africa': 'CAF',
    'Democratic Republic of the Congo': 'CAF',

    # AFC (Asia)
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC', 'China PR': 'AFC', 'Iraq': 'AFC',
    'Uzbekistan': 'AFC', 'Jordan': 'AFC',

    # OFC (Oceanía)
    'New Zealand': 'OFC',
}

confederacion_cod = {'UEFA': 1, 'CONMEBOL': 2, 'CONCACAF': 3, 'CAF': 4, 'AFC': 5, 'OFC': 6}

todos_equipos['confederacion'] = todos_equipos['equipo'].map(confederaciones).fillna('UEFA')
todos_equipos['confederacion_cod'] = todos_equipos['confederacion'].map(confederacion_cod)

# ─────────────────────────────────────────────
# 10. VARIABLE: INTERÉS EN GOOGLE TRENDS
# ─────────────────────────────────────────────
print("📊 Obteniendo datos de Google Trends...")

try:
    from pytrends.request import TrendReq
    import time

    pytrends = TrendReq(hl='en-US', tz=360)

    # Equipos clasificados al Mundial 2026 (ajusta según clasificados oficiales)
    equipos_2026 = [
        'United States', 'Mexico', 'Canada', 'Argentina', 'Brazil',
        'France', 'England', 'Spain', 'Germany', 'Portugal',
        'Morocco', 'Japan', 'South Korea', 'Australia', 'Netherlands',
        'Belgium', 'Croatia', 'Switzerland', 'Senegal', 'Uruguay'
    ]

    trends_data = {}

    # Google Trends solo admite 5 keywords por llamada
    for i in range(0, len(equipos_2026), 5):
        batch = [f"{eq} national football team" for eq in equipos_2026[i:i+5]]
        try:
            pytrends.build_payload(batch, timeframe='2025-09-01 2026-04-01', geo='')
            df_trends = pytrends.interest_over_time()
            if not df_trends.empty:
                for j, eq in enumerate(equipos_2026[i:i+5]):
                    col = batch[j]
                    if col in df_trends.columns:
                        trends_data[eq] = round(df_trends[col].mean(), 1)
            time.sleep(2)  # esperar para no bloquear la API
        except Exception as e:
            print(f"    ⚠️  Error en batch {i}: {e}")

    todos_equipos['interes_google'] = todos_equipos['equipo'].map(trends_data).fillna(10)
    print(f"  ✅ Google Trends obtenido para {len(trends_data)} equipos")

except ImportError:
    print("  ⚠️  pytrends no instalado. Usando valores por defecto.")
    print("      Instala con: pip install pytrends")
    # Valores estimados como fallback
    interes_estimado = {
        'Brazil': 85, 'Argentina': 82, 'France': 78, 'England': 75,
        'Spain': 70, 'Germany': 68, 'Portugal': 65, 'United States': 60,
        'Mexico': 72, 'Morocco': 55, 'Japan': 52, 'Netherlands': 50,
    }
    todos_equipos['interes_google'] = todos_equipos['equipo'].map(interes_estimado).fillna(20)

except Exception as e:
    print(f"  ⚠️  Error en Google Trends: {e}. Usando valores por defecto.")
    todos_equipos['interes_google'] = 30

# ─────────────────────────────────────────────
# 11. VARIABLE OBJETIVO: GANÓ EL MUNDIAL (histórico)
# ─────────────────────────────────────────────
print("📊 Construyendo variable objetivo (ganó el mundial)...")

# Ganadores históricos del Mundial FIFA
ganadores_mundiales = [
    'Uruguay', 'Italy', 'France', 'Germany', 'Brazil',
    'England', 'Argentina', 'Spain', 'West Germany'
]

todos_equipos['gano'] = todos_equipos['equipo'].apply(
    lambda e: 1 if e in ganadores_mundiales else 0
)

print(f"\n  Equipos con al menos 1 mundial: {todos_equipos['gano'].sum()}")

# ─────────────────────────────────────────────
# 12. GUARDAR DATASET FINAL
# ─────────────────────────────────────────────
columnas_modelo = [
    'equipo', 'racha_reciente', 'goles_favor_avg', 'goles_contra_avg',
    'diferencia_goles', 'veces_penales', 'gano_penales_pct',
    'ranking_fifa', 'es_local', 'confederacion', 'confederacion_cod',
    'interes_google', 'gano'
]

df_final = todos_equipos[columnas_modelo].copy()

# Imputacion final para evitar NaN en modelado y analisis estadistico.
num_cols = [
    'racha_reciente', 'goles_favor_avg', 'goles_contra_avg', 'diferencia_goles',
    'veces_penales', 'gano_penales_pct', 'ranking_fifa', 'es_local',
    'confederacion_cod', 'interes_google', 'gano'
]
for col in num_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    if df_final[col].isna().all():
        df_final[col] = 0
    else:
        df_final[col].fillna(df_final[col].median(), inplace=True)

df_final.to_csv('dataset_modelo.csv', index=False)

print(f"\n✅ Dataset final guardado: dataset_modelo.csv")
print(f"   Filas: {len(df_final)} | Columnas: {len(df_final.columns)}")
print(f"\n📋 Vista previa:")
print(df_final.head(10).to_string())
print(f"\n📊 Estadísticas descriptivas:")
print(df_final.describe().to_string())
