"""
visualizaciones.py
==================
Genera todas las gráficas del proyecto:
  1. Heatmap de correlaciones
  2. Feature importance
  3. Top 20 favoritos al mundial
  4. Distribución de probabilidades por confederación
  5. Comparativa: ganadores vs no ganadores
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Estilo general
plt.rcParams['figure.facecolor'] = '#0a0a1a'
plt.rcParams['axes.facecolor']   = '#0a0a1a'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'
plt.rcParams['axes.edgecolor']   = '#333355'

GOLD   = '#FFD700'
BLUE   = '#1a3a6b'
WHITE  = '#f0f0f0'
SILVER = '#C0C0C0'

print("🎨 Generando visualizaciones...")

# Cargar datos
df       = pd.read_csv('dataset_modelo.csv')
feat_imp = pd.read_csv('feature_importance.csv')
proba    = pd.read_csv('probabilidades_ganador_2026.csv')
corr_df  = pd.read_csv('resultados_correlaciones.csv')

features = [
    'racha_reciente', 'goles_favor_avg', 'goles_contra_avg',
    'diferencia_goles', 'gano_penales_pct', 'ranking_fifa',
    'es_local', 'confederacion_cod', 'interes_google'
]

# ─────────────────────────────────────────────
# GRÁFICA 1: HEATMAP DE CORRELACIONES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0a0a1a')

X = df[features + ['gano']].copy()
X['ranking_fifa'] = 1 / (X['ranking_fifa'] + 1)
corr_matrix = X.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='#1a1a2e',
    ax=ax, annot_kws={'size': 9, 'color': 'white'}
)

ax.set_title('Correlación entre Variables\nProyecto dataGol — Mundial 2026',
             fontsize=14, color=GOLD, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('grafica_correlaciones.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.close()
print("  ✅ grafica_correlaciones.png")

# ─────────────────────────────────────────────
# GRÁFICA 2: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0a0a1a')

feat_sorted = feat_imp.sort_values('importancia', ascending=True)
colores = [GOLD if i == len(feat_sorted) - 1 else SILVER if i >= len(feat_sorted) - 3 else '#4a6fa5'
           for i in range(len(feat_sorted))]

bars = ax.barh(feat_sorted['variable'], feat_sorted['importancia'],
               color=colores, edgecolor='none', height=0.65)

for bar, val in zip(bars, feat_sorted['importancia']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color='white', fontsize=9)

ax.set_xlabel('Importancia relativa', color=WHITE, fontsize=10)
ax.set_title('¿Qué Variables Predicen Mejor al Ganador?\nRandom Forest — Feature Importance',
             fontsize=13, color=GOLD, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, feat_sorted['importancia'].max() * 1.15)

plt.tight_layout()
plt.savefig('grafica_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.close()
print("  ✅ grafica_feature_importance.png")

# ─────────────────────────────────────────────
# GRÁFICA 3: TOP 20 FAVORITOS
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#0a0a1a')

top20 = proba.head(20).sort_values('proba_final', ascending=True)

colores_top = []
for i, row in top20.iterrows():
    if row.name == top20.index[-1]:
        colores_top.append(GOLD)
    elif row.name in top20.index[-3:]:
        colores_top.append('#FFA500')
    else:
        colores_top.append('#1a5276')

bars = ax.barh(top20['equipo'], top20['proba_final'] * 100,
               color=colores_top, edgecolor='none', height=0.7)

for bar, val in zip(bars, top20['proba_final'] * 100):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color='white', fontsize=9, fontweight='bold')

ax.set_xlabel('Probabilidad de Ganar el Mundial (%)', color=WHITE, fontsize=11)
ax.set_title('Top 20 Favoritos al Mundial 2026\nModelo Ensemble (Regresión Logística + Random Forest)',
             fontsize=13, color=GOLD, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, top20['proba_final'].max() * 100 * 1.15)

plt.tight_layout()
plt.savefig('grafica_top20_favoritos.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.close()
print("  ✅ grafica_top20_favoritos.png")

# ─────────────────────────────────────────────
# GRÁFICA 4: PROBABILIDAD POR CONFEDERACIÓN
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0a0a1a')

df_conf = proba.copy()
if 'confederacion' not in df_conf.columns:
    df_conf = df_conf.merge(df[['equipo', 'confederacion']], on='equipo', how='left')
df_conf = df_conf.dropna(subset=['confederacion'])
conf_avg = df_conf.groupby('confederacion')['proba_final'].mean().sort_values(ascending=False) * 100

colores_conf = [GOLD, '#FFA500', '#4a6fa5', '#2e86ab', '#1b4f72', '#7fb3d3'][:len(conf_avg)]
bars = ax.bar(conf_avg.index, conf_avg.values, color=colores_conf, edgecolor='none', width=0.6)

for bar, val in zip(bars, conf_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f'{val:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')

ax.set_ylabel('Probabilidad promedio (%)', color=WHITE, fontsize=11)
ax.set_title('Probabilidad Promedio por Confederación\n¿Qué zona del mundo tiene más chances?',
             fontsize=13, color=GOLD, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('grafica_por_confederacion.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.close()
print("  ✅ grafica_por_confederacion.png")

# ─────────────────────────────────────────────
# GRÁFICA 5: P-VALUES (significancia)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0a0a1a')

corr_sorted = corr_df.sort_values('p_value', ascending=True)
colores_pval = ['#27ae60' if p < 0.05 else '#e74c3c' for p in corr_sorted['p_value']]

bars = ax.barh(corr_sorted['variable'], corr_sorted['p_value'],
               color=colores_pval, edgecolor='none', height=0.65)

# Línea de significancia
ax.axvline(x=0.05, color=GOLD, linestyle='--', linewidth=1.5, label='p = 0.05 (umbral significancia)')

for bar, val in zip(bars, corr_sorted['p_value']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color='white', fontsize=9)

sig_patch   = mpatches.Patch(color='#27ae60', label='Significativa (p < 0.05)')
nosig_patch = mpatches.Patch(color='#e74c3c', label='No significativa (p ≥ 0.05)')
ax.legend(handles=[sig_patch, nosig_patch],
          facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white', fontsize=9)

ax.set_xlabel('P-value', color=WHITE, fontsize=11)
ax.set_title('Significancia Estadística de las Variables\n(menor p-value = más importante para el modelo)',
             fontsize=13, color=GOLD, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('grafica_pvalues.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.close()
print("  ✅ grafica_pvalues.png")

print("\n✅ Todas las gráficas generadas correctamente.")
print("   Archivos PNG listos para el informe/presentación.")
