"""
modelo_prediccion.py
====================
Entrena y evalúa dos modelos:
  1. Regresión Logística — interpretable, con probabilidades
  2. Random Forest — más preciso, con feature importance

Genera:
  - probabilidades_ganador_2026.csv  → equipos ordenados por probabilidad
  - metricas_modelos.csv             → comparativa de performance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🏆 MODELO PREDICTIVO — MUNDIAL 2026")
print("=" * 60)

# ─────────────────────────────────────────────
# CARGAR Y PREPARAR
# ─────────────────────────────────────────────
df = pd.read_csv('dataset_modelo.csv')

features = [
    'racha_reciente', 'goles_favor_avg', 'goles_contra_avg',
    'diferencia_goles', 'gano_penales_pct', 'ranking_fifa',
    'es_local', 'confederacion_cod', 'interes_google'
]

X = df[features].copy()
y = df['gano'].copy()

X = X.replace([np.inf, -np.inf], np.nan)
for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    if X[col].isna().all():
        X[col] = 0
    else:
        X[col].fillna(X[col].median(), inplace=True)

# El ranking FIFA: menor número = mejor equipo → lo invertimos
X['ranking_fifa'] = 1 / (X['ranking_fifa'] + 1)

# Escalar para regresión logística
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print(f"\n📋 Dataset: {len(df)} equipos | {len(features)} features")
print(f"   Equipos con mundial ganado: {y.sum()} | Sin mundial: {(y==0).sum()}")

# ─────────────────────────────────────────────
# 1. REGRESIÓN LOGÍSTICA
# ─────────────────────────────────────────────
print("\n\n🔵 MODELO 1: REGRESIÓN LOGÍSTICA")
print("-" * 40)

lr = LogisticRegression(
    class_weight='balanced',  # importante: dataset desbalanceado
    max_iter=1000,
    random_state=42,
    C=0.5
)

# Cross-validation estratificada (mejor para datasets pequeños)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

print(f"  AUC-ROC (cross-val 5-fold): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print(f"  Scores por fold: {[round(s, 3) for s in cv_scores_lr]}")

# Entrenar con todos los datos para predicción
lr.fit(X_scaled, y)
proba_lr = lr.predict_proba(X_scaled)[:, 1]

df['proba_logistica'] = proba_lr

# ─────────────────────────────────────────────
# 2. RANDOM FOREST
# ─────────────────────────────────────────────
print("\n\n🟢 MODELO 2: RANDOM FOREST")
print("-" * 40)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

print(f"  AUC-ROC (cross-val 5-fold): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"  Scores por fold: {[round(s, 3) for s in cv_scores_rf]}")

# Entrenar con todos los datos
rf.fit(X, y)
proba_rf = rf.predict_proba(X)[:, 1]
df['proba_rf'] = proba_rf

# ─────────────────────────────────────────────
# 3. PROBABILIDAD COMBINADA (ensemble)
# ─────────────────────────────────────────────
# Promedio ponderado: más peso al que tiene mejor AUC
peso_lr = cv_scores_lr.mean()
peso_rf = cv_scores_rf.mean()
total   = peso_lr + peso_rf

df['proba_final'] = (
    (peso_lr / total) * df['proba_logistica'] +
    (peso_rf / total) * df['proba_rf']
)

# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
print("\n\n📊 IMPORTANCIA DE VARIABLES (Random Forest)")
print("-" * 40)

importancias = pd.Series(rf.feature_importances_, index=features)
importancias_sorted = importancias.sort_values(ascending=False)

for feat, imp in importancias_sorted.items():
    barra = "█" * int(imp * 50)
    print(f"  {feat:<25} {barra} {imp:.4f}")

# ─────────────────────────────────────────────
# 5. RANKING DE FAVORITOS PARA 2026
# ─────────────────────────────────────────────
print("\n\n🏆 TOP 20 FAVORITOS AL MUNDIAL 2026")
print("-" * 50)

ranking_prediccion = df[['equipo', 'proba_final', 'proba_logistica', 'proba_rf', 'confederacion']].copy()
ranking_prediccion = ranking_prediccion.sort_values('proba_final', ascending=False).reset_index(drop=True)
ranking_prediccion.index += 1  # empezar en 1

ranking_prediccion['proba_final_%']    = (ranking_prediccion['proba_final']    * 100).round(1).astype(str) + '%'
ranking_prediccion['proba_logistica_%'] = (ranking_prediccion['proba_logistica'] * 100).round(1).astype(str) + '%'
ranking_prediccion['proba_rf_%']       = (ranking_prediccion['proba_rf']       * 100).round(1).astype(str) + '%'

top20 = ranking_prediccion.head(20)[['equipo', 'confederacion', 'proba_final_%', 'proba_logistica_%', 'proba_rf_%']]
top20.columns = ['Equipo', 'Confederación', 'Prob. Final', 'Regr. Logística', 'Random Forest']
print(top20.to_string())

# ─────────────────────────────────────────────
# 6. GUARDAR RESULTADOS
# ─────────────────────────────────────────────
ranking_prediccion.to_csv('probabilidades_ganador_2026.csv', index=True)

metricas = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Random Forest', 'Ensemble (combinado)'],
    'AUC-ROC (cv)': [
        f"{cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}",
        f"{cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}",
        'N/A (promedio ponderado)'
    ]
})
metricas.to_csv('metricas_modelos.csv', index=False)

importancias_df = importancias_sorted.reset_index()
importancias_df.columns = ['variable', 'importancia']
importancias_df.to_csv('feature_importance.csv', index=False)

print(f"\n\n✅ Modelos entrenados y guardados.")
print("   Archivos:")
print("   - probabilidades_ganador_2026.csv")
print("   - metricas_modelos.csv")
print("   - feature_importance.csv")
print(f"\n🎯 PREDICCIÓN: el favorito según el modelo es → {ranking_prediccion.iloc[0]['equipo']}")
