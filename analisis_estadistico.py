"""
analisis_estadistico.py
=======================
Análisis estadístico completo del dataset.
Esto es lo que diferencia un proyecto de Big Data de uno básico:
  - Matriz de correlación
  - Test de significancia (p-values)
  - Odds ratios (para regresión logística)
  - Interpretación automática de resultados
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📐 ANÁLISIS ESTADÍSTICO — MUNDIAL 2026")
print("=" * 60)

# ─────────────────────────────────────────────
# CARGAR DATOS
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
        X[col] = X[col].fillna(X[col].median())
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

# Normalizar ranking (menor rank = mejor → invertir)
X['ranking_fifa'] = 1 / (X['ranking_fifa'] + 1)

# ─────────────────────────────────────────────
# 1. CORRELACIONES CON LA VARIABLE OBJETIVO
# ─────────────────────────────────────────────
print("\n📊 1. CORRELACIÓN DE VARIABLES CON 'GANÓ MUNDIAL'")
print("-" * 50)

correlaciones = []
for col in features:
    valid = pd.DataFrame({"x": X[col], "y": y}).dropna()
    if valid["x"].nunique() <= 1:
        corr, pval = 0.0, 1.0
    else:
        corr, pval = stats.pointbiserialr(valid["x"], valid["y"])
    correlaciones.append({
        'variable': col,
        'correlacion': round(corr, 4),
        'p_value': round(pval, 4),
        'significativa': '✅ SÍ' if pval < 0.05 else '❌ NO'
    })

df_corr = pd.DataFrame(correlaciones).sort_values('correlacion', ascending=False)
print(df_corr.to_string(index=False))

# ─────────────────────────────────────────────
# 2. REGRESIÓN LOGÍSTICA CON STATSMODELS (p-values)
# ─────────────────────────────────────────────
print("\n\n📊 2. REGRESIÓN LOGÍSTICA — P-VALUES COMPLETOS")
print("-" * 50)

X_sm = sm.add_constant(X)

try:
    modelo_logit = sm.Logit(y, X_sm).fit(method='bfgs', maxiter=200, disp=False)
    summary = modelo_logit.summary2()
    print(modelo_logit.summary())

    # Extraer tabla de coeficientes
    tabla = summary.tables[1]
    print("\n\n📋 INTERPRETACIÓN AUTOMÁTICA:")
    print("-" * 50)

    for var in features:
        if var in modelo_logit.pvalues.index:
            pval = modelo_logit.pvalues[var]
            coef = modelo_logit.params[var]
            odds = np.exp(coef)

            if pval < 0.01:
                sig = "🔴 MUY significativa (p < 0.01)"
            elif pval < 0.05:
                sig = "🟡 Significativa (p < 0.05)"
            elif pval < 0.10:
                sig = "🟠 Marginalmente significativa (p < 0.10)"
            else:
                sig = "⚪ No significativa (p > 0.10)"

            direccion = "↑ positivo" if coef > 0 else "↓ negativo"

            print(f"\n  {var}")
            print(f"    Coeficiente: {coef:.4f} ({direccion})")
            print(f"    Odds Ratio:  {odds:.4f}")
            print(f"    P-value:     {pval:.4f}  → {sig}")

except Exception as e:
    print(f"⚠️  Error en Logit completo: {e}")
    print("Intentando con menos variables...")

    # Usar solo las numéricas sin multicolinealidad
    X_simple = X[['racha_reciente', 'diferencia_goles', 'ranking_fifa', 'interes_google']].copy()
    X_sm2 = sm.add_constant(X_simple)
    modelo_logit2 = sm.Logit(y, X_sm2).fit(disp=False)
    print(modelo_logit2.summary())

# ─────────────────────────────────────────────
# 3. TEST CHI-CUADRADO PARA VARIABLES CATEGÓRICAS
# ─────────────────────────────────────────────
print("\n\n📊 3. TEST CHI-CUADRADO — VARIABLES CATEGÓRICAS")
print("-" * 50)

# Es local
tabla_local = pd.crosstab(df['es_local'], df['gano'])
chi2, p, dof, _ = stats.chi2_contingency(tabla_local)
print(f"\n  es_local vs gano:")
print(f"  Chi²={chi2:.4f}, p={p:.4f} → {'✅ Significativa' if p < 0.05 else '❌ No significativa'}")

# Confederación
tabla_conf = pd.crosstab(df['confederacion'], df['gano'])
chi2, p, dof, _ = stats.chi2_contingency(tabla_conf)
print(f"\n  confederacion vs gano:")
print(f"  Chi²={chi2:.4f}, p={p:.4f} → {'✅ Significativa' if p < 0.05 else '❌ No significativa'}")
print(f"\n  Mundiales ganados por confederación:")
print(df.groupby('confederacion')['gano'].sum().sort_values(ascending=False).to_string())

# ─────────────────────────────────────────────
# 4. ANÁLISIS DESCRIPTIVO POR GRUPO
# ─────────────────────────────────────────────
print("\n\n📊 4. ESTADÍSTICAS: GANADORES vs NO GANADORES")
print("-" * 50)

ganadores    = df[df['gano'] == 1]
no_ganadores = df[df['gano'] == 0]

vars_numericas = ['racha_reciente', 'diferencia_goles', 'ranking_fifa', 'interes_google', 'goles_favor_avg']

comparativa = pd.DataFrame({
    'Ganadores (media)':     ganadores[vars_numericas].mean().round(3),
    'No ganadores (media)':  no_ganadores[vars_numericas].mean().round(3),
})
comparativa['Diferencia'] = (comparativa['Ganadores (media)'] - comparativa['No ganadores (media)']).round(3)
print(comparativa.to_string())

# ─────────────────────────────────────────────
# 5. GUARDAR RESULTADOS
# ─────────────────────────────────────────────
df_corr.to_csv('resultados_correlaciones.csv', index=False)
comparativa.to_csv('resultados_comparativa_grupos.csv')

print("\n\n✅ Análisis estadístico completado.")
print("   Archivos guardados:")
print("   - resultados_correlaciones.csv")
print("   - resultados_comparativa_grupos.csv")
