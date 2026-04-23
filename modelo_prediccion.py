"""
modelo_prediccion.py
====================
Entrena y evalúa tres modelos:
  1. Regresión Logística — interpretable, con probabilidades
  2. Random Forest       — preciso, con feature importance
  3. XGBoost             — gradient boosting, alta precisión

Genera:
  - probabilidades_ganador_2026.csv   → equipos ordenados por probabilidad
  - metricas_modelos.csv              → comparativa legacy (compatibilidad)
  - comparativa_modelos.csv           → comparativa completa con 3 modelos
  - feature_importance.csv            → importancia de variables (RF)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    print("⚠️  XGBoost no instalado. Ejecuta: pip install xgboost")
    XGBOOST_OK = False

print("=" * 65)
print("  🏆 MODELO PREDICTIVO — MUNDIAL 2026  (v2: +XGBoost)")
print("=" * 65)

# ─────────────────────────────────────────────
# CARGAR Y PREPARAR DATOS
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
    X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

# Ranking FIFA invertido: menor número de posición → mayor valor de feature
X['ranking_fifa'] = 1 / (X['ranking_fifa'] + 1)

# Escalar para Regresión Logística
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

n_pos = int(y.sum())
n_neg = int((y == 0).sum())
print(f"\n  Dataset: {len(df)} equipos | {len(features)} features")
print(f"  Ganadores históricos: {n_pos} | Sin mundial: {n_neg}")
print(f"  Desbalance: 1:{n_neg // n_pos if n_pos else 'N/A'}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# 1. REGRESIÓN LOGÍSTICA
# ─────────────────────────────────────────────
print("\n\n  🔵 MODELO 1: REGRESIÓN LOGÍSTICA")
print("  " + "─" * 45)

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=0.5
)

cv_scores_lr = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
lr.fit(X_scaled, y)
proba_lr = lr.predict_proba(X_scaled)[:, 1]
df['proba_logistica'] = proba_lr

print(f"  AUC-ROC (5-fold cv): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print(f"  Scores por fold:     {[round(s, 4) for s in cv_scores_lr]}")

# ─────────────────────────────────────────────
# 2. RANDOM FOREST
# ─────────────────────────────────────────────
print("\n\n  🟢 MODELO 2: RANDOM FOREST")
print("  " + "─" * 45)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
rf.fit(X, y)
proba_rf = rf.predict_proba(X)[:, 1]
df['proba_rf'] = proba_rf

print(f"  AUC-ROC (5-fold cv): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"  Scores por fold:     {[round(s, 4) for s in cv_scores_rf]}")

# ─────────────────────────────────────────────
# 3. XGBOOST
# ─────────────────────────────────────────────
print("\n\n  🟠 MODELO 3: XGBOOST")
print("  " + "─" * 45)

if XGBOOST_OK:
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,   # maneja el desbalance de clases
        random_state=42,
        eval_metric='auc',
        verbosity=0,
        use_label_encoder=False
    )

    cv_scores_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='roc_auc')
    xgb.fit(X, y)
    proba_xgb = xgb.predict_proba(X)[:, 1]
    df['proba_xgb'] = proba_xgb

    print(f"  AUC-ROC (5-fold cv): {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
    print(f"  Scores por fold:     {[round(s, 4) for s in cv_scores_xgb]}")
    print(f"  scale_pos_weight:    {scale_pos:.1f}  (ratio neg/pos)")
else:
    cv_scores_xgb = np.array([0.0])
    df['proba_xgb'] = df['proba_rf']   # fallback
    print("  ⚠️  XGBoost no disponible — usando RF como fallback")

# ─────────────────────────────────────────────
# 4. ENSEMBLE PONDERADO (LR + RF + XGBoost)
# ─────────────────────────────────────────────
# Pesos proporcionales al AUC de cada modelo
peso_lr  = cv_scores_lr.mean()
peso_rf  = cv_scores_rf.mean()
peso_xgb = cv_scores_xgb.mean() if XGBOOST_OK else 0.0
total    = peso_lr + peso_rf + peso_xgb

df['proba_final'] = (
    (peso_lr  / total) * df['proba_logistica'] +
    (peso_rf  / total) * df['proba_rf']         +
    (peso_xgb / total) * df['proba_xgb']
)

w_lr  = round(peso_lr  / total * 100, 1)
w_rf  = round(peso_rf  / total * 100, 1)
w_xgb = round(peso_xgb / total * 100, 1)
print(f"\n  📐 Pesos del ensemble:")
print(f"     LR: {w_lr}% | RF: {w_rf}% | XGBoost: {w_xgb}%")

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
print("\n\n  📊 IMPORTANCIA DE VARIABLES (Random Forest)")
print("  " + "─" * 45)

importancias = pd.Series(rf.feature_importances_, index=features)
importancias = importancias.sort_values(ascending=False)

for feat, imp in importancias.items():
    barra = "█" * int(imp * 50)
    print(f"  {feat:<25} {barra} {imp:.4f}")

# ─────────────────────────────────────────────
# 6. RANKING DE FAVORITOS 2026
# ─────────────────────────────────────────────
print("\n\n  🏆 TOP 20 FAVORITOS AL MUNDIAL 2026")
print("  " + "─" * 60)

ranking = df[['equipo', 'proba_final', 'proba_logistica',
              'proba_rf', 'proba_xgb', 'confederacion']].copy()
ranking = ranking.sort_values('proba_final', ascending=False).reset_index(drop=True)
ranking.index += 1

ranking['proba_final_%']     = (ranking['proba_final']     * 100).round(1).astype(str) + '%'
ranking['proba_logistica_%'] = (ranking['proba_logistica'] * 100).round(1).astype(str) + '%'
ranking['proba_rf_%']        = (ranking['proba_rf']        * 100).round(1).astype(str) + '%'
ranking['proba_xgb_%']       = (ranking['proba_xgb']       * 100).round(1).astype(str) + '%'

top20 = ranking.head(20)[[
    'equipo', 'confederacion', 'proba_final_%',
    'proba_logistica_%', 'proba_rf_%', 'proba_xgb_%'
]]
top20.columns = ['Equipo', 'Confederación', 'Prob. Final', 'Log. Reg.', 'Rnd. Forest', 'XGBoost']
print(top20.to_string())

# ─────────────────────────────────────────────
# 7. GUARDAR RESULTADOS
# ─────────────────────────────────────────────
ranking.to_csv('probabilidades_ganador_2026.csv', index=True)

# Comparativa de los 3 modelos (archivo principal)
comparativa = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Random Forest', 'XGBoost',
               'Ensemble (ponderado)'],
    'AUC_ROC_media': [
        round(cv_scores_lr.mean(),  4),
        round(cv_scores_rf.mean(),  4),
        round(cv_scores_xgb.mean(), 4) if XGBOOST_OK else None,
        None
    ],
    'AUC_ROC_std': [
        round(cv_scores_lr.std(),  4),
        round(cv_scores_rf.std(),  4),
        round(cv_scores_xgb.std(), 4) if XGBOOST_OK else None,
        None
    ],
    'AUC_ROC_display': [
        f"{cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}",
        f"{cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}",
        f"{cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}"
        if XGBOOST_OK else 'N/A',
        f'Promedio ponderado ({w_lr}% LR / {w_rf}% RF / {w_xgb}% XGB)'
    ],
    'Peso_Ensemble_%': [w_lr, w_rf, w_xgb, 100.0]
})
comparativa.to_csv('comparativa_modelos.csv', index=False)

# Archivo legacy para compatibilidad
metricas_legacy = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Random Forest',
               'XGBoost', 'Ensemble (combinado)'],
    'AUC-ROC (cv)': [
        f"{cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}",
        f"{cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}",
        f"{cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}"
        if XGBOOST_OK else 'N/A',
        f'Ponderado ({w_lr}/{w_rf}/{w_xgb}%)'
    ]
})
metricas_legacy.to_csv('metricas_modelos.csv', index=False)

importancias_df = importancias.reset_index()
importancias_df.columns = ['variable', 'importancia']
importancias_df.to_csv('feature_importance.csv', index=False)

print(f"\n\n  ✅ Modelos entrenados y guardados.")
print("     Archivos generados:")
print("     - probabilidades_ganador_2026.csv")
print("     - comparativa_modelos.csv")
print("     - metricas_modelos.csv")
print("     - feature_importance.csv")

print(f"\n  🎯 FAVORITO 2026 → {ranking.iloc[0]['equipo']}  "
      f"({ranking.iloc[0]['proba_final_%']})")

# ─────────────────────────────────────────────
# 8. TABLA COMPARATIVA FINAL
# ─────────────────────────────────────────────
print(f"\n\n  📈 COMPARATIVA DE MODELOS")
print("  " + "─" * 55)
for _, row in comparativa.iterrows():
    if row['AUC_ROC_media'] is not None:
        barra = "█" * int(row['AUC_ROC_media'] * 50)
        print(f"  {row['Modelo']:<25} {barra} {row['AUC_ROC_display']}")
    else:
        print(f"  {row['Modelo']:<25} {row['AUC_ROC_display']}")
