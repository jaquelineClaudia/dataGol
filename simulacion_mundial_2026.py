"""
simulacion_mundial_2026.py
==========================
Simula el Mundial 2026 completo 10.000 veces usando Monte Carlo:
  - Fase de grupos (72 partidos)
  - Round of 32, R16, Cuartos, Semis, Final
  - Calcula % campeon, finalista, semifinalista, etc.
  - Media, desviacion estandar e intervalos de confianza 95%

Fuentes de fuerza del equipo:
  - Puntos FIFA (ranking_fifa_masculino.csv)
  - Valor de mercado (valor_mercado_2026.csv)
  - Probabilidad del modelo ML (probabilidades_ganador_2026.csv)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_SIMS = 10_000
BASE_DIR = Path(__file__).resolve().parent

print("=" * 65)
print("  SIMULACION MONTE CARLO — MUNDIAL 2026")
print(f"  {N_SIMS:,} simulaciones completas del torneo")
print("=" * 65)


def read_csv_auto(filename):
    path = BASE_DIR / filename
    with open(path, 'rb') as f:
        signature = f.read(2)
    if signature == b'PK':
        return pd.read_excel(path)

    for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
    return pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')

# ─────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────
fixtures  = read_csv_auto('partidos.csv')
ranking   = read_csv_auto('ranking_fifa_masculino.csv')
mercado   = read_csv_auto('valor_mercado_2026.csv')
modelo    = read_csv_auto('probabilidades_ganador_2026.csv')

# ─────────────────────────────────────────────
# 2. NORMALIZAR NOMBRES DE EQUIPOS
# ─────────────────────────────────────────────
ALIASES = {
    'South Korea': 'Korea Republic',
    'Czech Republic': 'Czechia',
    'Iran': 'IR Iran',
    'Turkey': 'Turkiye',
    'Ivory Coast': "Cote d'Ivoire",
    'DR Congo': 'Congo DR',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'Curacao': 'Curacao',
    'Curaçao': 'Curacao',
    'Cape Verde': 'Cabo Verde',
    'USA': 'United States',
}

RANKING_ALIASES = {v: k for k, v in ALIASES.items()}
RANKING_ALIASES.update({
    'Korea Republic': 'South Korea',
    'Czechia': 'Czech Republic',
    'IR Iran': 'Iran',
    'Turkiye': 'Turkey',
    "Cote d'Ivoire": 'Ivory Coast',
    'Congo DR': 'DR Congo',
    'Bosnia and Herzegovina': 'Bosnia & Herzegovina',
    'United States': 'USA',
    'Cabo Verde': 'Cape Verde',
})


def norm(name):
    return ALIASES.get(str(name).strip(), str(name).strip())


# Equipos del fixture (nombres originales)
todos_equipos = sorted(set(
    fixtures['Equipo_1'].tolist() + fixtures['Equipo_2'].tolist()
))

# ─────────────────────────────────────────────
# 3. CONSTRUIR PUNTUACION DE FUERZA POR EQUIPO
# ─────────────────────────────────────────────
# --- FIFA ranking points ---
rk_map = {}
for _, row in ranking.iterrows():
    nombre_fix = RANKING_ALIASES.get(row['pais'], row['pais'])
    rk_map[nombre_fix] = float(row['puntos'])
    rk_map[row['pais']] = float(row['puntos'])

# --- Valor de mercado ---
vm_map = {}
for _, row in mercado.iterrows():
    vm_map[row['equipo']] = float(row['valor_mercado_millones_eur'])

# --- Probabilidad del modelo ML ---
ml_map = {}
for _, row in modelo.iterrows():
    ml_map[row['equipo']] = float(row['proba_final'])
    # alias inverso por si nombre difiere
    alt = RANKING_ALIASES.get(row['equipo'], row['equipo'])
    ml_map[alt] = float(row['proba_final'])

# Fallbacks globales
rk_median = np.median(list(rk_map.values())) if rk_map else 1000
vm_median = np.median(list(vm_map.values())) if vm_map else 100
ml_median = np.median(list(ml_map.values())) if ml_map else 0.03

def get_strength(equipo):
    """Puntuacion de fuerza combinada (0-1) para cada equipo."""
    rk  = rk_map.get(equipo, rk_median)
    vm  = vm_map.get(equipo, vm_median)
    ml  = ml_map.get(equipo, ml_median)
    return rk, vm, ml

# Calcular fuerza bruta para todos los equipos del fixture
strength_raw = {}
for eq in todos_equipos:
    rk, vm, ml = get_strength(eq)
    strength_raw[eq] = (rk, vm, ml)

# Normalizar cada dimension a [0,1]
rk_vals = np.array([strength_raw[e][0] for e in todos_equipos])
vm_vals = np.array([strength_raw[e][1] for e in todos_equipos])
ml_vals = np.array([strength_raw[e][2] for e in todos_equipos])

def minmax(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

rk_norm = minmax(rk_vals)
vm_norm = minmax(vm_vals)
ml_norm = minmax(ml_vals)

STRENGTH = {}
for i, eq in enumerate(todos_equipos):
    # Pesos: FIFA 40%, Mercado 30%, Modelo ML 30%
    STRENGTH[eq] = 0.40 * rk_norm[i] + 0.30 * vm_norm[i] + 0.30 * ml_norm[i]

print("\n📊 Fuerza de equipos (top 15):")
strength_sorted = sorted(STRENGTH.items(), key=lambda x: -x[1])
for eq, s in strength_sorted[:15]:
    barra = "█" * int(s * 30)
    print(f"  {eq:<25} {barra} {s:.4f}")

# ─────────────────────────────────────────────
# 4. FUNCION PROBABILIDAD DE PARTIDO
# ─────────────────────────────────────────────
def match_probs(eq_a, eq_b, knockout=False):
    """
    Devuelve (p_gana_A, p_empate, p_gana_B).
    En eliminatorias no hay empate en 90 min → empate va a penales.
    """
    sa = STRENGTH.get(eq_a, 0.5)
    sb = STRENGTH.get(eq_b, 0.5)

    # Diferencia de fuerza → probabilidad via logistica
    diff = sa - sb
    p_a_raw = 1 / (1 + np.exp(-8 * diff))   # 8 = sensibilidad

    # Probabilidad de empate (mayor cuando equipos son parejos)
    paridad = 1 - abs(diff) * 2
    p_draw = np.clip(0.22 * paridad + 0.05, 0.05, 0.30)

    if knockout:
        p_draw = 0.0   # no hay empate, va a penales directamente

    resto = 1 - p_draw
    p_a = resto * p_a_raw
    p_b = resto * (1 - p_a_raw)

    return p_a, p_draw, p_b


def simular_partido(eq_a, eq_b, knockout=False):
    """
    Retorna: 'A' si gana A, 'B' si gana B, 'D' si empate (solo fase grupos).
    En knockout, el empate se resuelve en penales (50/50 ajustado por fuerza).
    """
    pa, pd, pb = match_probs(eq_a, eq_b, knockout=knockout)
    resultado = np.random.choice(['A', 'D', 'B'], p=[pa, pd, pb])

    if resultado == 'D' and knockout:
        # Penales: equipo mas fuerte tiene leve ventaja
        sa = STRENGTH.get(eq_a, 0.5)
        sb = STRENGTH.get(eq_b, 0.5)
        p_pen_a = 0.5 + 0.1 * (sa - sb)
        p_pen_a = np.clip(p_pen_a, 0.35, 0.65)
        resultado = 'A' if np.random.random() < p_pen_a else 'B'

    return resultado


def simular_goles(eq_a, eq_b):
    """Simula marcador aproximado para diferencia de goles en fase grupos."""
    sa = STRENGTH.get(eq_a, 0.5)
    sb = STRENGTH.get(eq_b, 0.5)
    media_total = 2.6
    lambda_a = media_total * (0.5 + 0.6 * (sa - sb))
    lambda_b = media_total * (0.5 + 0.6 * (sb - sa))
    lambda_a = max(0.3, lambda_a)
    lambda_b = max(0.3, lambda_b)
    ga = np.random.poisson(lambda_a)
    gb = np.random.poisson(lambda_b)
    return ga, gb

# ─────────────────────────────────────────────
# 5. ESTRUCTURA DEL TORNEO
# ─────────────────────────────────────────────
# Grupos del fixture
GRUPOS = {}
for g in sorted(fixtures['Grupo'].unique()):
    df_g = fixtures[fixtures['Grupo'] == g]
    equipos = sorted(set(df_g['Equipo_1'].tolist() + df_g['Equipo_2'].tolist()))
    partidos = list(zip(df_g['Equipo_1'].tolist(), df_g['Equipo_2'].tolist()))
    GRUPOS[g] = {'equipos': equipos, 'partidos': partidos}

# Bracket Round of 32:
# 12 grupos pareados: A-B, C-D, E-F, G-H, I-J, K-L
# Cada par genera 2 cruces: 1°A vs 2°B  y  1°B vs 2°A
# Los 8 mejores 3ros se emparejan entre si (4 cruces)
PARES_GRUPOS = [('A','B'), ('C','D'), ('E','F'), ('G','H'), ('I','J'), ('K','L')]

# Bracket fijo de R32 → R16 → QF → SF → Final
# Indice de los 16 partidos de R32 dentro del bracket general
BRACKET_R32 = [
    # Par AB
    ('1A', '2B'),   # match 0
    ('1B', '2A'),   # match 1
    # Par CD
    ('1C', '2D'),   # match 2
    ('1D', '2C'),   # match 3
    # Par EF
    ('1E', '2F'),   # match 4
    ('1F', '2E'),   # match 5
    # Par GH
    ('1G', '2H'),   # match 6
    ('1H', '2G'),   # match 7
    # Par IJ
    ('1I', '2J'),   # match 8
    ('1J', '2I'),   # match 9
    # Par KL
    ('1K', '2L'),   # match 10
    ('1L', '2K'),   # match 11
    # 8 mejores 3ros (ordenados por puntos) vs entre si
    ('3A', '3B'),   # match 12  (mejor 3ro vs 2do mejor 3ro)
    ('3C', '3D'),   # match 13
    ('3E', '3F'),   # match 14
    ('3G', '3H'),   # match 15
]

# R16: pares de ganadores de R32 (indices de R32)
BRACKET_R16 = [(0,2), (1,12), (4,6), (5,13), (8,10), (9,14), (3,7), (11,15)]
# QF: pares de ganadores de R16
BRACKET_QF  = [(0,1), (2,3), (4,5), (6,7)]
# SF: pares de ganadores de QF
BRACKET_SF  = [(0,1), (2,3)]
# Final: ganadores de SF

# ─────────────────────────────────────────────
# 6. SIMULACION DE FASE DE GRUPOS
# ─────────────────────────────────────────────
def simular_grupo(grupo_id):
    """
    Simula todos los partidos de un grupo.
    Devuelve lista de equipos ordenada: [1°, 2°, 3°, 4°]
    """
    equipos  = GRUPOS[grupo_id]['equipos']
    partidos = GRUPOS[grupo_id]['partidos']

    stats = {eq: {'pts': 0, 'gf': 0, 'gc': 0, 'gd': 0} for eq in equipos}

    for eq_a, eq_b in partidos:
        ga, gb = simular_goles(eq_a, eq_b)

        if ga > gb:
            stats[eq_a]['pts'] += 3
        elif ga == gb:
            stats[eq_a]['pts'] += 1
            stats[eq_b]['pts'] += 1
        else:
            stats[eq_b]['pts'] += 3

        stats[eq_a]['gf'] += ga
        stats[eq_a]['gc'] += gb
        stats[eq_a]['gd'] += (ga - gb)
        stats[eq_b]['gf'] += gb
        stats[eq_b]['gc'] += ga
        stats[eq_b]['gd'] += (gb - ga)

    # Ordenar: puntos → dif. goles → goles favor → fuerza (desempate)
    clasificacion = sorted(
        equipos,
        key=lambda e: (
            stats[e]['pts'],
            stats[e]['gd'],
            stats[e]['gf'],
            STRENGTH.get(e, 0) + np.random.uniform(-0.05, 0.05),
        ),
        reverse=True,
    )

    return clasificacion, stats


# ─────────────────────────────────────────────
# 7. SIMULACION COMPLETA DE UN MUNDIAL
# ─────────────────────────────────────────────
def simular_mundial():
    """Una simulacion completa. Devuelve {equipo: puesto_final}."""
    resultados_grupo = {}
    stats_grupos = {}

    # --- Fase de grupos ---
    for g in sorted(GRUPOS.keys()):
        clasi, stats = simular_grupo(g)
        resultados_grupo[g] = clasi   # [1°, 2°, 3°, 4°]
        stats_grupos[g] = stats

    # --- Mejores 3ros ---
    terceros = []
    for g, clasi in resultados_grupo.items():
        eq_3ro = clasi[2]
        pts = stats_grupos[g][eq_3ro]['pts']
        gd  = stats_grupos[g][eq_3ro]['gd']
        gf  = stats_grupos[g][eq_3ro]['gf']
        terceros.append((eq_3ro, pts, gd, gf, g))

    terceros_sorted = sorted(terceros, key=lambda x: (x[1], x[2], x[3]), reverse=True)
    mejores_8_3ros  = [t[0] for t in terceros_sorted[:8]]

    # Slot de 3ros en el bracket (3A=mejor, 3B=2do, ...)
    slots_3ros = {f'3{chr(65+i)}': mejores_8_3ros[i] for i in range(8)}

    # Resolver slots del bracket R32
    def resolver_slot(slot):
        if slot.startswith('3'):
            return slots_3ros.get(slot, None)
        pos = int(slot[0]) - 1      # '1' → 0, '2' → 1
        grp = slot[1]               # 'A', 'B', ...
        clasi = resultados_grupo.get(grp, [])
        return clasi[pos] if len(clasi) > pos else None

    # --- Eliminatorias ---
    posicion = defaultdict(lambda: 'Fase Grupos')

    def jugar_ronda(emparejamientos, equipos_ronda, nombre_ronda):
        """Simula una ronda eliminatoria. Devuelve lista de ganadores."""
        ganadores = []
        for i, (slot_a, slot_b) in enumerate(emparejamientos):
            eq_a = equipos_ronda[i][0]
            eq_b = equipos_ronda[i][1]
            if eq_a is None or eq_b is None:
                ganador = eq_a or eq_b
            else:
                res = simular_partido(eq_a, eq_b, knockout=True)
                ganador  = eq_a if res == 'A' else eq_b
                perdedor = eq_b if res == 'A' else eq_a
                posicion[perdedor] = nombre_ronda
            ganadores.append(ganador)
        return ganadores

    # R32
    equipos_r32 = []
    for slot_a, slot_b in BRACKET_R32:
        equipos_r32.append((resolver_slot(slot_a), resolver_slot(slot_b)))

    ganadores_r32 = []
    for eq_a, eq_b in equipos_r32:
        if eq_a is None and eq_b is None:
            ganadores_r32.append(None)
        elif eq_a is None:
            ganadores_r32.append(eq_b)
        elif eq_b is None:
            ganadores_r32.append(eq_a)
        else:
            res = simular_partido(eq_a, eq_b, knockout=True)
            gan = eq_a if res == 'A' else eq_b
            per = eq_b if res == 'A' else eq_a
            posicion[per] = 'R32'
            ganadores_r32.append(gan)

    # R16
    equipos_r16 = [(ganadores_r32[i], ganadores_r32[j]) for i, j in BRACKET_R16]
    ganadores_r16 = []
    for eq_a, eq_b in equipos_r16:
        if eq_a is None and eq_b is None:
            ganadores_r16.append(None)
        elif eq_a is None:
            ganadores_r16.append(eq_b)
        elif eq_b is None:
            ganadores_r16.append(eq_a)
        else:
            res = simular_partido(eq_a, eq_b, knockout=True)
            gan = eq_a if res == 'A' else eq_b
            per = eq_b if res == 'A' else eq_a
            posicion[per] = 'R16'
            ganadores_r16.append(gan)

    # Cuartos
    equipos_qf = [(ganadores_r16[i], ganadores_r16[j]) for i, j in BRACKET_QF]
    ganadores_qf = []
    for eq_a, eq_b in equipos_qf:
        if eq_a is None and eq_b is None:
            ganadores_qf.append(None)
        elif eq_a is None:
            ganadores_qf.append(eq_b)
        elif eq_b is None:
            ganadores_qf.append(eq_a)
        else:
            res = simular_partido(eq_a, eq_b, knockout=True)
            gan = eq_a if res == 'A' else eq_b
            per = eq_b if res == 'A' else eq_a
            posicion[per] = 'Cuartos de Final'
            ganadores_qf.append(gan)

    # Semis
    equipos_sf = [(ganadores_qf[i], ganadores_qf[j]) for i, j in BRACKET_SF]
    ganadores_sf = []
    for eq_a, eq_b in equipos_sf:
        if eq_a is None and eq_b is None:
            ganadores_sf.append(None)
        elif eq_a is None:
            ganadores_sf.append(eq_b)
        elif eq_b is None:
            ganadores_sf.append(eq_a)
        else:
            res = simular_partido(eq_a, eq_b, knockout=True)
            gan = eq_a if res == 'A' else eq_b
            per = eq_b if res == 'A' else eq_a
            posicion[per] = 'Semifinal'
            ganadores_sf.append(gan)

    # Final
    eq_a, eq_b = ganadores_sf[0], ganadores_sf[1]
    if eq_a is None and eq_b is None:
        campeon = None
    elif eq_a is None:
        campeon = eq_b
        posicion[eq_b] = 'Campeon'
    elif eq_b is None:
        campeon = eq_a
        posicion[eq_a] = 'Campeon'
    else:
        res = simular_partido(eq_a, eq_b, knockout=True)
        campeon  = eq_a if res == 'A' else eq_b
        finalista = eq_b if res == 'A' else eq_a
        posicion[campeon]   = 'Campeon'
        posicion[finalista] = 'Final'

    return posicion


# ─────────────────────────────────────────────
# 8. MONTE CARLO — 10.000 SIMULACIONES
# ─────────────────────────────────────────────
print(f"\n⚙️  Ejecutando {N_SIMS:,} simulaciones...\n")

ETAPAS = ['Fase Grupos', 'R32', 'R16', 'Cuartos de Final', 'Semifinal', 'Final', 'Campeon']
ETAPA_NUM = {e: i for i, e in enumerate(ETAPAS)}

conteo = defaultdict(lambda: defaultdict(int))   # conteo[equipo][etapa]

for sim in range(N_SIMS):
    if (sim + 1) % 2000 == 0:
        print(f"  Simulacion {sim+1:,} / {N_SIMS:,}...")

    resultado = simular_mundial()
    for eq in todos_equipos:
        etapa = resultado.get(eq, 'Fase Grupos')
        conteo[eq][etapa] += 1

print("  ✅ Simulaciones completadas.\n")

# ─────────────────────────────────────────────
# 9. CALCULAR ESTADISTICAS
# ─────────────────────────────────────────────
rows = []
for eq in todos_equipos:
    c = conteo[eq]
    total = N_SIMS

    p_campeon  = c.get('Campeon', 0) / total
    p_final    = (c.get('Campeon', 0) + c.get('Final', 0)) / total
    p_semi     = sum(c.get(e, 0) for e in ['Campeon','Final','Semifinal']) / total
    p_cuartos  = sum(c.get(e, 0) for e in ['Campeon','Final','Semifinal','Cuartos de Final']) / total
    p_r16      = sum(c.get(e, 0) for e in ['Campeon','Final','Semifinal','Cuartos de Final','R16']) / total
    p_r32      = sum(c.get(e, 0) for e in ETAPAS if e != 'Fase Grupos') / total

    # Estadisticas bootstrap via formula binomial
    # Error estandar para una proporcion: sqrt(p*(1-p)/n)
    se_campeon = np.sqrt(p_campeon * (1 - p_campeon) / total)
    ic_low  = max(0, p_campeon - 1.96 * se_campeon)
    ic_high = min(1, p_campeon + 1.96 * se_campeon)

    # Posicion promedio (numerica)
    posicion_num = []
    for etapa, cnt in c.items():
        posicion_num.extend([ETAPA_NUM.get(etapa, 0)] * cnt)
    posicion_num.extend([0] * (total - sum(c.values())))

    media_pos  = np.mean(posicion_num)
    std_pos    = np.std(posicion_num)

    rows.append({
        'Equipo'        : eq,
        'Grupo'         : next((g for g, d in GRUPOS.items() if eq in d['equipos']), '?'),
        'Fuerza'        : round(STRENGTH.get(eq, 0), 4),
        'P_Campeon_%'   : round(p_campeon * 100, 2),
        'P_Final_%'     : round(p_final * 100, 2),
        'P_Semifinal_%' : round(p_semi * 100, 2),
        'P_Cuartos_%'   : round(p_cuartos * 100, 2),
        'P_R16_%'       : round(p_r16 * 100, 2),
        'P_Clasifica_%' : round(p_r32 * 100, 2),
        'IC95_Bajo_%'   : round(ic_low * 100, 2),
        'IC95_Alto_%'   : round(ic_high * 100, 2),
        'Media_Ronda'   : round(media_pos, 3),
        'Std_Ronda'     : round(std_pos, 3),
        'N_Campeones'   : c.get('Campeon', 0),
    })

df_result = pd.DataFrame(rows).sort_values('P_Campeon_%', ascending=False).reset_index(drop=True)
df_result.index += 1

# ─────────────────────────────────────────────
# 10. IMPRIMIR RESULTADOS
# ─────────────────────────────────────────────
print("=" * 65)
print("  TOP 20 FAVORITOS AL TITULO — MUNDIAL 2026")
print("  (% basado en 10.000 simulaciones del torneo completo)")
print("=" * 65)
print(f"\n{'#':<4} {'Equipo':<22} {'Grupo':<6} {'Campeon':>8} {'Final':>7} {'Semi':>7} {'Clasif':>8} {'IC 95%':>16}")
print("-" * 80)
for _, row in df_result.head(20).iterrows():
    ic = f"[{row['IC95_Bajo_%']:.1f}%-{row['IC95_Alto_%']:.1f}%]"
    print(
        f"{_:<4} {row['Equipo']:<22} {row['Grupo']:<6} "
        f"{row['P_Campeon_%']:>7.1f}%"
        f"{row['P_Final_%']:>7.1f}%"
        f"{row['P_Semifinal_%']:>7.1f}%"
        f"{row['P_Clasifica_%']:>8.1f}%"
        f"  {ic}"
    )

print("\n")
print("=" * 65)
print("  PROBABILIDAD DE CLASIFICAR A R32 POR GRUPO")
print("=" * 65)
for g in sorted(GRUPOS.keys()):
    print(f"\n  Grupo {g}:")
    grupo_df = df_result[df_result['Grupo'] == g].sort_values('P_Clasifica_%', ascending=False)
    for _, row in grupo_df.iterrows():
        barra = "█" * int(row['P_Clasifica_%'] / 5)
        print(f"    {row['Equipo']:<25} {barra:<20} {row['P_Clasifica_%']:>5.1f}%  (campeon: {row['P_Campeon_%']:.1f}%)")

# ─────────────────────────────────────────────
# 11. GUARDAR RESULTADOS
# ─────────────────────────────────────────────
df_result.to_csv(BASE_DIR / 'simulacion_monte_carlo_2026.csv', index=True)

# Resumen ejecutivo
resumen = df_result[['Equipo','Grupo','Fuerza',
                      'P_Campeon_%','IC95_Bajo_%','IC95_Alto_%',
                      'P_Final_%','P_Semifinal_%','P_Cuartos_%',
                      'P_R16_%','P_Clasifica_%',
                      'Media_Ronda','Std_Ronda']].copy()
resumen.to_csv(BASE_DIR / 'resumen_simulacion_2026.csv', index=True)

print("\n")
print("=" * 65)
print("  ESTADISTICAS DEL CAMPEON PREDICHO")
print("=" * 65)
campeon_predicho = df_result.iloc[0]
print(f"\n  Favorito #1: {campeon_predicho['Equipo']} (Grupo {campeon_predicho['Grupo']})")
print(f"  Probabilidad de campeon:    {campeon_predicho['P_Campeon_%']:.1f}%")
print(f"  Intervalo de confianza 95%: [{campeon_predicho['IC95_Bajo_%']:.1f}%, {campeon_predicho['IC95_Alto_%']:.1f}%]")
print(f"  Probabilidad de final:      {campeon_predicho['P_Final_%']:.1f}%")
print(f"  Probabilidad de semifinal:  {campeon_predicho['P_Semifinal_%']:.1f}%")
print(f"  Fuerza del equipo:          {campeon_predicho['Fuerza']:.4f}")
print(f"  Ronda promedio alcanzada:   {campeon_predicho['Media_Ronda']:.2f} ± {campeon_predicho['Std_Ronda']:.2f}")
print(f"  Fue campeon en:             {campeon_predicho['N_Campeones']:,} de {N_SIMS:,} simulaciones")

print(f"\n✅ Resultados guardados en:")
print(f"   - simulacion_monte_carlo_2026.csv  (tabla completa)")
print(f"   - resumen_simulacion_2026.csv      (resumen ejecutivo)")
