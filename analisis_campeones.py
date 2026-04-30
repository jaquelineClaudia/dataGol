"""
analisis_campeones.py
====================
Análisis profundo de los últimos 5 campeones del Mundial FIFA (2006, 2010, 2014, 2018, 2022)
Extrae estadísticas clave: goles, penales, racha, ranking FIFA.
Identifica patrones comunes del ADN del campeón.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
print("📂 Cargando archivos...")

results    = pd.read_csv('results.csv')
goalscorers = pd.read_csv('goalscorers.csv')
shootouts   = pd.read_csv('shootouts.csv')
ranking_fifa = pd.read_csv('ranking_fifa_masculino.csv')
equipos_2026 = pd.read_excel('equipos_mundial.csv', sheet_name='mundial')

results['date'] = pd.to_datetime(results['date'])
shootouts['date'] = pd.to_datetime(shootouts['date'])
ranking_fifa['fecha_publicacion'] = pd.to_datetime(ranking_fifa['fecha_publicacion'], errors='coerce')

# ─────────────────────────────────────────────
# DEFINICIÓN DE CAMPEONES Y MUNDIALES
# ─────────────────────────────────────────────
campeones_data = {
    2006: {'campeon': 'Italy', 'inicio': '2006-06-09', 'fin': '2006-07-09'},
    2010: {'campeon': 'Spain', 'inicio': '2010-06-11', 'fin': '2010-07-11'},
    2014: {'campeon': 'Germany', 'inicio': '2014-06-12', 'fin': '2014-07-13'},
    2018: {'campeon': 'France', 'inicio': '2018-06-14', 'fin': '2018-07-15'},
    2022: {'campeon': 'Argentina', 'inicio': '2022-11-21', 'fin': '2022-12-18'},
}

# ─────────────────────────────────────────────
# FUNCIÓN: Extrae stats de un campeón en su mundial
# ─────────────────────────────────────────────
def extraer_stats_campeon(campeon, inicio_str, fin_str, year):
    """Extrae goles, penales, racha y ranking FIFA de un campeón durante su mundial."""
    
    inicio = pd.to_datetime(inicio_str)
    fin = pd.to_datetime(fin_str)
    
    # Partidos del equipo en el mundial (como local y visitante)
    partidos_local = results[
        (results['home_team'] == campeon) &
        (results['date'] >= inicio) &
        (results['date'] <= fin) &
        (results['tournament'].str.contains('FIFA World Cup', na=False, case=False))
    ]
    
    partidos_visita = results[
        (results['away_team'] == campeon) &
        (results['date'] >= inicio) &
        (results['date'] <= fin) &
        (results['tournament'].str.contains('FIFA World Cup', na=False, case=False))
    ]
    
    partidos_mundial = pd.concat([partidos_local, partidos_visita]).sort_values('date')
    
    if len(partidos_mundial) == 0:
        print(f"⚠️  No se encontraron partidos para {campeon} en {year}")
        return None
    
    # Goles a favor y en contra
    goles_favor = partidos_local['home_score'].sum() + partidos_visita['away_score'].sum()
    goles_contra = partidos_local['away_score'].sum() + partidos_visita['home_score'].sum()
    
    n_partidos = len(partidos_mundial)
    goles_favor_promedio = goles_favor / n_partidos if n_partidos > 0 else 0
    goles_contra_promedio = goles_contra / n_partidos if n_partidos > 0 else 0
    
    # Penales
    penales_como_local = shootouts[
        (shootouts['home_team'] == campeon) &
        (shootouts['date'] >= inicio) &
        (shootouts['date'] <= fin)
    ]
    penales_como_visita = shootouts[
        (shootouts['away_team'] == campeon) &
        (shootouts['date'] >= inicio) &
        (shootouts['date'] <= fin)
    ]
    
    fue_a_penales = 1 if (len(penales_como_local) > 0 or len(penales_como_visita) > 0) else 0
    
    # Racha: % de victorias en últimos 10 partidos ANTES del mundial
    fecha_inicio_mundial = pd.to_datetime(inicio_str)
    fecha_inicio_racha = fecha_inicio_mundial - timedelta(days=365)  # últimos 12 meses antes del mundial
    
    partidos_previos = results[
        (((results['home_team'] == campeon) | (results['away_team'] == campeon)) &
         (results['date'] >= fecha_inicio_racha) &
         (results['date'] < fecha_inicio_mundial))
    ].sort_values('date').tail(10)
    
    victorias_previas = 0
    for _, p in partidos_previos.iterrows():
        if p['home_team'] == campeon and p['home_score'] > p['away_score']:
            victorias_previas += 1
        elif p['away_team'] == campeon and p['away_score'] > p['home_score']:
            victorias_previas += 1
    
    racha_reciente = (victorias_previas / len(partidos_previos)) if len(partidos_previos) > 0 else 0
    
    # Ranking FIFA más cercano a la fecha del mundial
    fecha_mundial_aprox = pd.to_datetime(f"{year}-06-15")
    fecha_mundial_aprox = fecha_mundial_aprox.tz_localize('UTC') if fecha_mundial_aprox.tz is None else fecha_mundial_aprox
    ranking_cercano = ranking_fifa[
        (ranking_fifa['pais'] == campeon) &
        (ranking_fifa['fecha_publicacion'].dt.tz_localize(None) <= fecha_mundial_aprox.tz_localize(None))
    ].sort_values('fecha_publicacion', ascending=False)
    
    ranking_en_mundial = ranking_cercano.iloc[0]['posicion'] if len(ranking_cercano) > 0 else np.nan
    
    return {
        'ano': year,
        'campeon': campeon,
        'n_partidos': n_partidos,
        'goles_favor': goles_favor,
        'goles_contra': goles_contra,
        'goles_favor_promedio': round(goles_favor_promedio, 2),
        'goles_contra_promedio': round(goles_contra_promedio, 2),
        'diferencia_goles_promedio': round(goles_favor_promedio - goles_contra_promedio, 2),
        'fue_a_penales': fue_a_penales,
        'racha_previo': round(racha_reciente, 2),
        'ranking_fifa_mundial': int(ranking_en_mundial) if pd.notna(ranking_en_mundial) else np.nan
    }

# ─────────────────────────────────────────────
# EXTRACCIÓN DE DATOS
# ─────────────────────────────────────────────
print("\n🏆 Extrayendo estadísticas de campeones...")

resultados_campeones = []
for year, data in campeones_data.items():
    stats = extraer_stats_campeon(
        data['campeon'],
        data['inicio'],
        data['fin'],
        year
    )
    if stats:
        resultados_campeones.append(stats)
        print(f"  ✅ {year} - {data['campeon']}: {stats['goles_favor_promedio']} GF, "
              f"{stats['goles_contra_promedio']} GC, racha={stats['racha_previo']}")

df_campeones = pd.DataFrame(resultados_campeones)

# ─────────────────────────────────────────────
# ANÁLISIS DE PATRONES
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("  📊 ANÁLISIS: ADN DEL CAMPEÓN (Últimos 5 mundiales)")
print("="*70)

print(f"\n🎯 ESTADÍSTICAS PROMEDIO DEL CAMPEÓN HISTÓRICO:")
print(f"  Goles a favor promedio:          {df_campeones['goles_favor_promedio'].mean():.2f}")
print(f"  Goles en contra promedio:        {df_campeones['goles_contra_promedio'].mean():.2f}")
print(f"  Diferencia de goles promedio:    {df_campeones['diferencia_goles_promedio'].mean():.2f}")
print(f"  % de campeones que fue a penales: {df_campeones['fue_a_penales'].sum()}/{len(df_campeones)}")
print(f"  Racha promedio (antes mundial):  {df_campeones['racha_previo'].mean():.1%}")
print(f"  Ranking FIFA promedio en mundial: {df_campeones['ranking_fifa_mundial'].mean():.0f}")

print(f"\n📈 COMPARATIVA POR AÑO:")
print(df_campeones[['ano', 'campeon', 'goles_favor_promedio', 'goles_contra_promedio', 
                     'fue_a_penales', 'racha_previo', 'ranking_fifa_mundial']].to_string(index=False))

# ─────────────────────────────────────────────
# GUARDAR DATOS CAMPEONES
# ─────────────────────────────────────────────
df_campeones.to_csv('analisis_campeones.csv', index=False)
print(f"\n✅ Datos de campeones guardados en: analisis_campeones.csv")

# ─────────────────────────────────────────────
# ANÁLISIS: JUGADOR TOP 2026
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("  🌟 ANÁLISIS: JUGADORES TOP EN EQUIPOS CLASIFICADOS 2026")
print("="*70)

# Lista de equipos clasificados 2026 con jugadores estrella
jugadores_top_2026 = {
    'Francia': 1,
    'Brasil': 1,
    'Argentina': 1,
    'Inglaterra': 1,
    'Portugal': 1,
    'España': 1,
    'Alemania': 1,
    'Países Bajos': 1,
}

# Mapeo español-inglés
mapa_esp_eng = {
    'Francia': 'France',
    'Brasil': 'Brazil',
    'Argentina': 'Argentina',
    'Inglaterra': 'England',
    'Portugal': 'Portugal',
    'España': 'Spain',
    'Alemania': 'Germany',
    'Países Bajos': 'Netherlands',
}

# Crear dataframe de 48 equipos con columna tiene_jugador_top
equipos_2026_lista = equipos_2026['Seleccion'].dropna().astype(str).str.strip().tolist()

resultado_jugadores = []
for equipo_esp in equipos_2026_lista:
    equipo_eng = mapa_esp_eng.get(equipo_esp, equipo_esp)
    tiene_top = jugadores_top_2026.get(equipo_esp, 0)
    resultado_jugadores.append({
        'equipo_es': equipo_esp,
        'equipo_en': equipo_eng,
        'tiene_jugador_top': tiene_top
    })

df_jugadores = pd.DataFrame(resultado_jugadores)
df_jugadores.to_csv('analisis_jugador_top.csv', index=False)

print(f"\n✅ Datos de jugadores top guardados en: analisis_jugador_top.csv")
print(f"   Equipos con jugador top: {df_jugadores['tiene_jugador_top'].sum()}/48")

print("\n" + "="*70)
print("  ✅ ANÁLISIS COMPLETADO")
print("="*70)
print(f"\n📋 Archivos generados:")
print(f"  1. analisis_campeones.csv (5 campeones con sus stats)")
print(f"  2. analisis_jugador_top.csv (48 equipos clasificados con flag jugador top)")
