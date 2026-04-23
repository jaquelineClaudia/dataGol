"""
backtesting.py
==============
Valida el modelo predictivo contra resultados históricos conocidos:
  - Entrena con datos hasta 2014, predice ganador del Mundial 2018
  - Entrena con datos hasta 2018, predice ganador del Mundial 2022

Guarda: backtesting_resultados.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
GANADORES_HISTORICOS = [
    'Uruguay', 'Italy', 'France', 'Germany', 'Brazil',
    'England', 'Argentina', 'Spain', 'West Germany'
]

GANADORES_REALES = {
    2018: 'France',
    2022: 'Argentina'
}

EQUIPOS_2018 = [
    'Russia', 'Saudi Arabia', 'Egypt', 'Uruguay',
    'Portugal', 'Spain', 'Morocco', 'Iran',
    'France', 'Australia', 'Peru', 'Denmark',
    'Argentina', 'Iceland', 'Croatia', 'Nigeria',
    'Brazil', 'Switzerland', 'Costa Rica', 'Serbia',
    'Germany', 'Mexico', 'Sweden', 'South Korea',
    'Belgium', 'Panama', 'Tunisia', 'England',
    'Poland', 'Senegal', 'Colombia', 'Japan'
]

EQUIPOS_2022 = [
    'Qatar', 'Ecuador', 'Senegal', 'Netherlands',
    'England', 'Iran', 'United States', 'Wales',
    'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
    'France', 'Australia', 'Denmark', 'Tunisia',
    'Spain', 'Costa Rica', 'Germany', 'Japan',
    'Belgium', 'Canada', 'Morocco', 'Croatia',
    'Brazil', 'Serbia', 'Switzerland', 'Cameroon',
    'Portugal', 'Ghana', 'Uruguay', 'South Korea'
]

SEDES = {
    2018: ['Russia'],
    2022: ['Qatar']
}

# Rankings FIFA reales antes de cada Mundial (posición → número menor = mejor)
RANKING_2018 = {
    'Germany': 1, 'Brazil': 2, 'Belgium': 3, 'Portugal': 4,
    'Argentina': 5, 'Spain': 6, 'France': 7, 'England': 8,
    'Colombia': 9, 'Uruguay': 10, 'Croatia': 11, 'Denmark': 12,
    'Switzerland': 13, 'Peru': 14, 'Russia': 15, 'Mexico': 16,
    'Iceland': 17, 'Poland': 18, 'Egypt': 19, 'Serbia': 20,
    'Morocco': 21, 'Nigeria': 22, 'Iran': 23, 'South Korea': 24,
    'Japan': 25, 'Senegal': 26, 'Australia': 27, 'Sweden': 28,
    'Costa Rica': 29, 'Panama': 30, 'Saudi Arabia': 31, 'Tunisia': 32,
}

RANKING_2022 = {
    'Brazil': 1, 'Belgium': 2, 'Argentina': 3, 'France': 4,
    'England': 5, 'Spain': 6, 'Portugal': 7, 'Netherlands': 8,
    'Denmark': 9, 'Germany': 10, 'Mexico': 11, 'United States': 12,
    'Croatia': 13, 'Uruguay': 14, 'Switzerland': 15, 'Morocco': 16,
    'Senegal': 17, 'Poland': 18, 'Australia': 19, 'Japan': 20,
    'Iran': 21, 'Serbia': 22, 'Ecuador': 23, 'Canada': 24,
    'Wales': 25, 'Qatar': 26, 'Tunisia': 27, 'South Korea': 28,
    'Ghana': 29, 'Costa Rica': 30, 'Cameroon': 31, 'Saudi Arabia': 32
}

CONFEDERACIONES = {
    'Germany': 'UEFA', 'France': 'UEFA', 'Spain': 'UEFA', 'Italy': 'UEFA',
    'England': 'UEFA', 'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Belgium': 'UEFA',
    'Croatia': 'UEFA', 'Switzerland': 'UEFA', 'Denmark': 'UEFA', 'Serbia': 'UEFA',
    'Poland': 'UEFA', 'Austria': 'UEFA', 'Sweden': 'UEFA', 'Iceland': 'UEFA',
    'Scotland': 'UEFA', 'Wales': 'UEFA', 'Ukraine': 'UEFA', 'Greece': 'UEFA',
    'Russia': 'UEFA', 'West Germany': 'UEFA', 'Czech Republic': 'UEFA',
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Peru': 'CONMEBOL',
    'Ecuador': 'CONMEBOL', 'Paraguay': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Costa Rica': 'CONCACAF', 'Honduras': 'CONCACAF', 'Panama': 'CONCACAF',
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Ghana': 'CAF', 'Nigeria': 'CAF',
    'Cameroon': 'CAF', 'Tunisia': 'CAF', 'Egypt': 'CAF', 'Algeria': 'CAF',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'Qatar': 'AFC',
    'New Zealand': 'OFC',
}

CONF_COD = {'UEFA': 1, 'CONMEBOL': 2, 'CONCACAF': 3, 'CAF': 4, 'AFC': 5, 'OFC': 6}

FEATURES = [
    'racha_reciente', 'goles_favor_avg', 'goles_contra_avg',
    'diferencia_goles', 'gano_penales_pct', 'ranking_fifa',
    'es_local', 'confederacion_cod', 'interes_google'
]

# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────
def read_csv_auto(path):
    for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.read_csv(path, on_bad_lines='skip')


def calcular_racha(equipo, df, fecha_corte, n=10):
    """% victorias en últimos n partidos antes de la fecha de corte."""
    partidos = df[
        ((df['home_team'] == equipo) | (df['away_team'] == equipo)) &
        (df['date'] < fecha_corte)
    ].sort_values('date').tail(n)
    if len(partidos) == 0:
        return 0.0
    wins = sum(
        1 for _, p in partidos.iterrows()
        if (p['home_team'] == equipo and p['home_score'] > p['away_score']) or
           (p['away_team'] == equipo and p['away_score'] > p['home_score'])
    )
    return wins / len(partidos)


def calcular_goles(equipo, mundiales_previos):
    """Promedio de goles en mundiales históricos previos."""
    local  = mundiales_previos[mundiales_previos['home_team'] == equipo]
    visita = mundiales_previos[mundiales_previos['away_team'] == equipo]
    gf = list(local['home_score']) + list(visita['away_score'])
    gc = list(local['away_score']) + list(visita['home_score'])
    return (round(np.mean(gf), 3) if gf else 0.0,
            round(np.mean(gc), 3) if gc else 0.0)


def calcular_penales(equipo, shootouts):
    """% victorias en tandas de penales (histórico)."""
    partic = shootouts[
        (shootouts['home_team'] == equipo) | (shootouts['away_team'] == equipo)
    ]
    if len(partic) == 0:
        return 0, 0.5
    victorias = len(partic[partic['winner'] == equipo])
    return len(partic), victorias / len(partic)


def construir_dataset(equipos, results_hist, mundiales_prev, shootouts,
                      fecha_corte, ranking_dict, sedes):
    """Construye DataFrame de features para una lista de equipos."""
    rows = []
    for eq in equipos:
        racha     = calcular_racha(eq, results_hist, fecha_corte)
        gf, gc    = calcular_goles(eq, mundiales_prev)
        _, pen_pct = calcular_penales(eq, shootouts)
        rank      = ranking_dict.get(eq, 50)
        conf      = CONFEDERACIONES.get(eq, 'UEFA')
        rows.append({
            'equipo'          : eq,
            'racha_reciente'  : racha,
            'goles_favor_avg' : gf,
            'goles_contra_avg': gc,
            'diferencia_goles': gf - gc,
            'gano_penales_pct': pen_pct,
            'ranking_fifa'    : 1.0 / (rank + 1),   # invertido: mayor = mejor equipo
            'es_local'        : 1 if eq in sedes else 0,
            'confederacion_cod': CONF_COD.get(conf, 1),
            'interes_google'  : 30.0,                # sin dato histórico → valor neutral
            'confederacion'   : conf,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────
print("=" * 65)
print("  ⏱️  BACKTESTING HISTÓRICO — MUNDIALES 2018 y 2022")
print("=" * 65)

results   = read_csv_auto('results.csv')
results['date'] = pd.to_datetime(results['date'])
shootouts = read_csv_auto('shootouts.csv')

mundiales_todos = results[
    results['tournament'].str.contains('FIFA World Cup', na=False, case=False)
].copy()

print(f"  Partidos internacionales cargados: {len(results):,}")
print(f"  Partidos de Mundial (histórico):   {len(mundiales_todos):,}")


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE BACKTEST
# ─────────────────────────────────────────────
def backtest_mundial(año_pred):
    """Entrena con datos hasta año_pred-4, predice ganador del Mundial año_pred."""
    print(f"\n{'─' * 65}")
    print(f"  📅 Prediciendo Mundial {año_pred}  "
          f"(entrenando con datos hasta {año_pred - 4})")
    print(f"{'─' * 65}")

    equipos_pred = EQUIPOS_2018 if año_pred == 2018 else EQUIPOS_2022
    ranking_dict = RANKING_2018 if año_pred == 2018 else RANKING_2022

    # Fechas exactas de inicio del torneo
    fecha_torneo = {2018: '2018-06-14', 2022: '2022-11-20'}
    fecha_corte  = pd.Timestamp(fecha_torneo[año_pred])
    año_corte    = año_pred - 4  # último WC conocido antes del que predecimos

    # Datos históricos previos al torneo
    mundiales_prev = mundiales_todos[mundiales_todos['date'].dt.year < año_pred].copy()
    results_hist   = results[results['date'] < fecha_corte].copy()

    print(f"  Mundiales usados para entrenar: "
          f"{sorted(mundiales_prev['date'].dt.year.unique()[-6:])}")
    print(f"  Partidos disponibles para racha: {len(results_hist):,}")

    # ── Dataset de entrenamiento ──
    equipos_hist = set(
        mundiales_prev['home_team'].tolist() + mundiales_prev['away_team'].tolist()
    )
    sedes_train = SEDES.get(año_corte, [])

    df_train = construir_dataset(
        list(equipos_hist), results_hist, mundiales_prev,
        shootouts, fecha_corte, ranking_dict, sedes_train
    )
    df_train['gano'] = df_train['equipo'].apply(
        lambda e: 1 if e in GANADORES_HISTORICOS else 0
    )

    # ── Dataset de predicción ──
    df_pred = construir_dataset(
        equipos_pred, results_hist, mundiales_prev,
        shootouts, fecha_corte, ranking_dict, SEDES.get(año_pred, [])
    )

    # Rellenar NaN con mediana del training
    mediana_train = df_train[FEATURES].median()
    X_train = df_train[FEATURES].fillna(mediana_train)
    y_train = df_train['gano']
    X_pred  = df_pred[FEATURES].fillna(mediana_train)

    print(f"  Equipos entrenamiento: {len(df_train):>4} | Campeones hist.: {y_train.sum()}")
    print(f"  Equipos a predecir:    {len(df_pred):>4}")

    # ── Entrenar modelos ──
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_pr_sc  = scaler.transform(X_pred)

    lr = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, C=0.5
    )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    )
    lr.fit(X_tr_sc, y_train)
    rf.fit(X_train, y_train)

    df_pred = df_pred.copy()
    df_pred['proba_lr']       = lr.predict_proba(X_pr_sc)[:, 1]
    df_pred['proba_rf']       = rf.predict_proba(X_pred)[:, 1]
    df_pred['proba_ensemble'] = 0.4 * df_pred['proba_lr'] + 0.6 * df_pred['proba_rf']
    df_pred = df_pred.sort_values('proba_ensemble', ascending=False).reset_index(drop=True)
    df_pred.index += 1

    # ── Evaluación ──
    ganador_real = GANADORES_REALES[año_pred]
    ganador_pred = df_pred.iloc[0]['equipo']

    rank_real = int(df_pred[df_pred['equipo'] == ganador_real].index[0]) \
        if ganador_real in df_pred['equipo'].values else 99
    prob_real = float(df_pred[df_pred['equipo'] == ganador_real]['proba_ensemble'].values[0]) \
        if ganador_real in df_pred['equipo'].values else 0.0

    acierto_top1 = ganador_pred == ganador_real
    acierto_top3 = ganador_real in df_pred.head(3)['equipo'].values
    acierto_top5 = ganador_real in df_pred.head(5)['equipo'].values

    # ── Imprimir resultado ──
    print(f"\n  TOP 10 PREDICCIONES para {año_pred}:")
    print(f"  {'#':>3}  {'Equipo':<22}  {'Conf.':<10}  {'Prob. Ensemble':>14}  {'LR':>8}  {'RF':>8}")
    print(f"  {'─'*75}")
    for i, row in df_pred.head(10).iterrows():
        marker = " ✅" if row['equipo'] == ganador_real else ""
        print(f"  {i:>3}  {row['equipo']:<22}  {row['confederacion']:<10}  "
              f"{row['proba_ensemble']:>13.1%}  "
              f"{row['proba_lr']:>7.1%}  {row['proba_rf']:>7.1%}{marker}")

    print(f"\n  {'Ganador real:':<30} {ganador_real}")
    print(f"  {'Predicción #1:':<30} {ganador_pred}")
    print(f"  {'Puesto del ganador real:':<30} #{rank_real}")
    print(f"  {'Prob. asignada al ganador:':<30} {prob_real:.1%}")
    print(f"  {'Acierto Top-1:':<30} {'✅ SÍ' if acierto_top1 else '❌ NO'}")
    print(f"  {'Acierto Top-3:':<30} {'✅ SÍ' if acierto_top3 else '❌ NO'}")
    print(f"  {'Acierto Top-5:':<30} {'✅ SÍ' if acierto_top5 else '❌ NO'}")

    top5 = df_pred.head(5)['equipo'].tolist()
    return {
        'Mundial'               : año_pred,
        'Datos_hasta'           : año_pred - 4,
        'Ganador_Real'          : ganador_real,
        'Prediccion_1'          : top5[0],
        'Prediccion_2'          : top5[1],
        'Prediccion_3'          : top5[2],
        'Prediccion_4'          : top5[3],
        'Prediccion_5'          : top5[4],
        'Puesto_Ganador_Real'   : rank_real,
        'Probabilidad_Ganador_%': round(prob_real * 100, 1),
        'Acierto_Top1'          : acierto_top1,
        'Acierto_Top3'          : acierto_top3,
        'Acierto_Top5'          : acierto_top5,
        'Prob_1_%'              : round(df_pred.iloc[0]['proba_ensemble'] * 100, 1),
        'Prob_2_%'              : round(df_pred.iloc[1]['proba_ensemble'] * 100, 1),
        'Prob_3_%'              : round(df_pred.iloc[2]['proba_ensemble'] * 100, 1),
    }


# ─────────────────────────────────────────────
# EJECUTAR BACKTESTS
# ─────────────────────────────────────────────
resultados = []
for año in [2018, 2022]:
    res = backtest_mundial(año)
    resultados.append(res)

# ─────────────────────────────────────────────
# RESUMEN COMPARATIVA
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  📊 RESUMEN COMPARATIVA DE BACKTESTING")
print(f"{'=' * 65}")

df_res = pd.DataFrame(resultados)

print(f"\n  {'Mundial':<10} {'Ganador Real':<15} {'Pred #1':<15} {'Pred #2':<15} {'Pred #3':<15} {'Puesto':<8} {'Prob%':<8}")
print(f"  {'─'*85}")
for _, r in df_res.iterrows():
    print(f"  {int(r.Mundial):<10} {r.Ganador_Real:<15} {r.Prediccion_1:<15} "
          f"{r.Prediccion_2:<15} {r.Prediccion_3:<15} "
          f"#{int(r.Puesto_Ganador_Real):<7} {r['Probabilidad_Ganador_%']}%")

acc_top1 = df_res['Acierto_Top1'].mean()
acc_top3 = df_res['Acierto_Top3'].mean()
acc_top5 = df_res['Acierto_Top5'].mean()

print(f"\n  PRECISIÓN GLOBAL (2 mundiales):")
print(f"    Top-1 accuracy: {acc_top1:.0%}  ({int(df_res['Acierto_Top1'].sum())}/2 mundiales acertados)")
print(f"    Top-3 accuracy: {acc_top3:.0%}  ({int(df_res['Acierto_Top3'].sum())}/2 mundiales acertados)")
print(f"    Top-5 accuracy: {acc_top5:.0%}  ({int(df_res['Acierto_Top5'].sum())}/2 mundiales acertados)")

# ─────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────
df_res.to_csv('backtesting_resultados.csv', index=False)
print(f"\n✅ Guardado: backtesting_resultados.csv")
print(f"   {len(df_res)} filas × {len(df_res.columns)} columnas")
