"""
dashboard.py
============
Dashboard interactivo del proyecto dataGol — Predictor del Mundial 2026.
Ejecutar con: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="dataGol — Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────
GOLD    = "#FFD700"
GOLD2   = "#FFA500"
SILVER  = "#C0C0C0"
BG_DARK = "#0D1117"
BG_MED  = "#161B22"
BG_CARD = "#21262D"
TEXT    = "#E6EDF3"
GREEN   = "#2EA043"
RED     = "#F85149"

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Fondo general */
  .stApp {{ background-color: {BG_DARK}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {BG_MED}; }}

  /* Título principal */
  .main-title {{
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, {GOLD}, {GOLD2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 0.3rem 0;
  }}
  .sub-title {{
    text-align: center;
    color: {SILVER};
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
  }}

  /* Tarjetas métricas */
  .metric-card {{
    background: {BG_CARD};
    border: 1px solid {GOLD}33;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }}
  .metric-val  {{ font-size: 2rem; font-weight: 800; color: {GOLD}; }}
  .metric-lbl  {{ font-size: 0.8rem; color: {SILVER}; margin-top: 2px; }}

  /* Tablas */
  .styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  .styled-table th {{
    background: {BG_MED};
    color: {GOLD};
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid {GOLD}55;
  }}
  .styled-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid #30363D;
    color: {TEXT};
  }}
  .styled-table tr:hover td {{ background: {BG_MED}; }}

  /* Barra de progreso personalizada */
  .bar-wrap {{ background: #21262D; border-radius: 6px; overflow: hidden; height: 22px; }}
  .bar-fill  {{ height: 100%; border-radius: 6px; display: flex;
               align-items: center; padding-left: 8px;
               font-size: 0.78rem; font-weight: 700; color: #000; }}

  /* Badges confederación */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.75rem; font-weight: 600; }}
  .UEFA      {{ background: #1F6FEB33; color: #58A6FF; }}
  .CONMEBOL  {{ background: #2EA04333; color: #56D364; }}
  .CONCACAF  {{ background: #D2993633; color: #E3B341; }}
  .CAF       {{ background: #8B4513aa; color: #FFA07A; }}
  .AFC       {{ background: #6E40C933; color: #BC8CFF; }}
  .OFC       {{ background: #FF6B6B33; color: #FF8E8E; }}

  /* Separador dorado */
  hr.gold {{ border: 0; border-top: 1px solid {GOLD}44; margin: 1rem 0; }}
  div[data-testid="stTabs"] button {{ color: {SILVER}; }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {GOLD}; border-bottom: 3px solid {GOLD};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS (con caché)
# ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def cargar_datos():
    archivos = {
        'probabilidades' : 'probabilidades_ganador_2026.csv',
        'feature_imp'    : 'feature_importance.csv',
        'comparativa'    : 'comparativa_modelos.csv',
        'metricas'       : 'metricas_modelos.csv',
        'backtesting'    : 'backtesting_resultados.csv',
        'simulacion'     : 'simulacion_monte_carlo_2026.csv',
    }
    data = {}
    for key, path in archivos.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data

data = cargar_datos()

# Columnas de probabilidad
def get_prob_df():
    df = data['probabilidades']
    if df is None:
        return pd.DataFrame()
    if 'proba_final' not in df.columns and 'proba_final_%' in df.columns:
        df['proba_final'] = df['proba_final_%'].str.replace('%', '').astype(float) / 100
    # Normalizar confederación
    df['confederacion'] = df.get('confederacion', 'UEFA')
    return df

prob_df = get_prob_df()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">⚽ dataGol — Predictor del Mundial 2026</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning · Monte Carlo · Backtesting · '
            'Análisis Estadístico</div>', unsafe_allow_html=True)
st.markdown('<hr class="gold">', unsafe_allow_html=True)

# ── Métricas rápidas en el header ──
if not prob_df.empty and data['simulacion'] is not None:
    sim_df = data['simulacion']
    favorito_ml  = prob_df.iloc[0]['equipo'] if len(prob_df) > 0 else "—"
    prob_fav_ml  = prob_df.iloc[0]['proba_final'] if len(prob_df) > 0 else 0

    sim_cols = [c for c in sim_df.columns if 'Campeon' in c or 'campeon' in c.lower()]
    if sim_cols:
        sim_sorted    = sim_df.sort_values(sim_cols[0], ascending=False)
        favorito_mc   = sim_sorted.iloc[0]['Equipo']
        prob_fav_mc   = sim_sorted.iloc[0][sim_cols[0]]
    else:
        favorito_mc, prob_fav_mc = "—", 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-val">{favorito_ml}</div>'
                    f'<div class="metric-lbl">🥇 Favorito (ML Ensemble)</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-val">{prob_fav_ml:.1%}</div>'
                    f'<div class="metric-lbl">Prob. de ser campeón (ML)</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-val">{favorito_mc}</div>'
                    f'<div class="metric-lbl">🎲 Favorito (Monte Carlo)</div></div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-val">{prob_fav_mc:.1f}%</div>'
                    f'<div class="metric-lbl">Prob. Monte Carlo (10K sims)</div></div>',
                    unsafe_allow_html=True)

st.markdown('<hr class="gold">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR: FILTROS
# ─────────────────────────────────────────────
st.sidebar.markdown(f"### ⚙️ Filtros")
st.sidebar.markdown("---")

confederaciones_disp = ['Todas']
if not prob_df.empty and 'confederacion' in prob_df.columns:
    confs_unicas = sorted(prob_df['confederacion'].dropna().unique().tolist())
    confederaciones_disp += confs_unicas

conf_sel = st.sidebar.selectbox("🌍 Confederación", confederaciones_disp)

n_equipos = st.sidebar.slider("🔢 Equipos a mostrar", min_value=5, max_value=48, value=20, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='color:{SILVER}; font-size:0.82rem;'>
<b>Metodología</b><br>
• Ensemble: LR + RF + XGBoost<br>
• Validación: AUC-ROC 5-fold CV<br>
• Simulación: 10.000 torneos<br>
• Backtesting: 2018 y 2022
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Favoritos 2026",
    "🎲 Monte Carlo",
    "📊 Feature Importance",
    "⏱️ Backtesting",
    "🔬 Comparativa Modelos"
])

# ════════════════════════════════════════════
# TAB 1: FAVORITOS 2026
# ════════════════════════════════════════════
with tab1:
    st.subheader(f"Top {n_equipos} Favoritos al Mundial 2026")
    st.caption("Probabilidades del ensemble (Regresión Logística + Random Forest + XGBoost)")

    if prob_df.empty:
        st.warning("Ejecuta `python modelo_prediccion.py` para generar las predicciones.")
    else:
        # Filtrar por confederación
        df_vis = prob_df.copy()
        if conf_sel != 'Todas':
            df_vis = df_vis[df_vis['confederacion'] == conf_sel]

        df_vis = df_vis.head(n_equipos).reset_index(drop=True)

        col_chart, col_table = st.columns([6, 4], gap="large")

        with col_chart:
            fig, ax = plt.subplots(figsize=(8, max(5, len(df_vis) * 0.45)))
            fig.patch.set_facecolor(BG_DARK)
            ax.set_facecolor(BG_DARK)

            equipos_plot = df_vis['equipo'].tolist()[::-1]
            probas_plot  = (df_vis['proba_final'] * 100).tolist()[::-1]
            confs_plot   = df_vis['confederacion'].tolist()[::-1]

            conf_colors = {
                'UEFA': '#1F6FEB', 'CONMEBOL': '#2EA043',
                'CONCACAF': '#E3B341', 'CAF': '#FFA07A',
                'AFC': '#BC8CFF', 'OFC': '#FF8E8E'
            }
            bar_colors = [conf_colors.get(c, GOLD) for c in confs_plot]

            bars = ax.barh(equipos_plot, probas_plot, color=bar_colors,
                           height=0.7, edgecolor='none')

            # Etiquetas de valor
            for bar, val in zip(bars, probas_plot):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', ha='left',
                        color=TEXT, fontsize=8.5, fontweight='bold')

            ax.set_xlim(0, max(probas_plot) * 1.18)
            ax.set_xlabel('Probabilidad de ser campeón (%)', color=SILVER, fontsize=9)
            ax.tick_params(colors=TEXT, labelsize=9)
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            ax.spines['left'].set_color('#30363D')
            ax.xaxis.grid(True, color='#30363D', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)

            # Leyenda confederaciones
            legend_patches = [
                mpatches.Patch(color=c, label=conf)
                for conf, c in conf_colors.items()
                if conf in confs_plot
            ]
            ax.legend(handles=legend_patches, loc='lower right',
                      facecolor=BG_MED, edgecolor='#30363D',
                      labelcolor=TEXT, fontsize=8)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_table:
            st.markdown("**Tabla de probabilidades**")
            # Construir tabla HTML estilizada
            rows_html = ""
            for i, row in df_vis.iterrows():
                medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"{i+1}.")
                conf  = row.get('confederacion', '')
                prob  = row['proba_final'] * 100
                prob_lr = row.get('proba_logistica', row['proba_final']) * 100
                prob_rf = row.get('proba_rf', row['proba_final']) * 100
                prob_xgb = row.get('proba_xgb', row['proba_final']) * 100

                bar_w = int(prob)
                bar_color = GOLD if i == 0 else ('#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#1F6FEB')

                rows_html += f"""
                <tr>
                  <td>{medal}</td>
                  <td><b>{row['equipo']}</b></td>
                  <td><span class="badge {conf}">{conf}</span></td>
                  <td>
                    <div class="bar-wrap">
                      <div class="bar-fill" style="width:{min(bar_w,100)}%;
                           background:linear-gradient(90deg,{bar_color},{GOLD2});">
                        {prob:.1f}%
                      </div>
                    </div>
                  </td>
                </tr>"""

            st.markdown(f"""
            <table class="styled-table">
              <thead><tr>
                <th>#</th><th>Equipo</th><th>Conf.</th><th>Probabilidad</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2: MONTE CARLO
# ════════════════════════════════════════════
with tab2:
    st.subheader("Simulación Monte Carlo — 10.000 torneos")
    st.caption("Probabilidad de alcanzar cada ronda basada en fortaleza FIFA + valor mercado + ML")

    if data['simulacion'] is None:
        st.warning("Ejecuta `python simulacion_mundial_2026.py` para generar la simulación.")
    else:
        sim = data['simulacion'].copy()

        # Detectar columna campeón
        camp_col = next((c for c in sim.columns if 'Campeon' in c), None)
        final_col = next((c for c in sim.columns if 'Final' in c and 'Campeon' not in c), None)
        semi_col  = next((c for c in sim.columns if 'Semifinal' in c), None)

        if camp_col:
            sim = sim.sort_values(camp_col, ascending=False)
            if conf_sel != 'Todas' and 'Confederacion' in sim.columns:
                sim = sim[sim['Confederacion'] == conf_sel]
            sim_top = sim.head(n_equipos)

            col_a, col_b = st.columns([6, 4], gap="large")
            with col_a:
                fig2, ax2 = plt.subplots(figsize=(8, max(5, len(sim_top) * 0.45)))
                fig2.patch.set_facecolor(BG_DARK)
                ax2.set_facecolor(BG_DARK)

                teams_mc = sim_top['Equipo'].tolist()[::-1]
                probs_mc  = sim_top[camp_col].tolist()[::-1]

                cmap_colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(teams_mc)))[::-1]
                bars2 = ax2.barh(teams_mc, probs_mc,
                                  color=cmap_colors, height=0.7, edgecolor='none')

                for bar, val in zip(bars2, probs_mc):
                    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                             f'{val:.2f}%', va='center', ha='left',
                             color=TEXT, fontsize=8.5, fontweight='bold')

                ax2.set_xlim(0, max(probs_mc) * 1.2)
                ax2.set_xlabel('P(Campeón) — % de 10.000 simulaciones', color=SILVER, fontsize=9)
                ax2.tick_params(colors=TEXT, labelsize=9)
                ax2.spines[['top', 'right', 'bottom']].set_visible(False)
                ax2.spines['left'].set_color('#30363D')
                ax2.xaxis.grid(True, color='#30363D', linestyle='--', alpha=0.5)
                ax2.set_axisbelow(True)
                ax2.set_title('Probabilidad de ser campeón (Monte Carlo)',
                              color=GOLD, fontsize=10, pad=10)

                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            with col_b:
                display_cols = ['Equipo']
                if camp_col:  display_cols.append(camp_col)
                if final_col: display_cols.append(final_col)
                if semi_col:  display_cols.append(semi_col)
                if 'IC95_Bajo_%' in sim.columns: display_cols += ['IC95_Bajo_%', 'IC95_Alto_%']

                sim_show = sim_top[display_cols].reset_index(drop=True)
                sim_show.index += 1

                rename_map = {
                    camp_col:     'P.Campeón %',
                    final_col:    'P.Final %',
                    semi_col:     'P.Semi %',
                    'IC95_Bajo_%': 'IC95 bajo',
                    'IC95_Alto_%': 'IC95 alto',
                }
                sim_show = sim_show.rename(columns={k: v for k, v in rename_map.items() if k})
                st.dataframe(sim_show, use_container_width=True, height=450)

# ════════════════════════════════════════════
# TAB 3: FEATURE IMPORTANCE
# ════════════════════════════════════════════
with tab3:
    st.subheader("Importancia de Variables — Random Forest")
    st.caption("Contribución de cada variable al poder predictivo del modelo")

    if data['feature_imp'] is None:
        st.warning("Ejecuta `python modelo_prediccion.py` para generar feature_importance.csv")
    else:
        fi = data['feature_imp'].copy()
        fi = fi.sort_values('importancia', ascending=False)

        nombres_legibles = {
            'ranking_fifa'    : 'Ranking FIFA',
            'interes_google'  : 'Interés Google Trends',
            'racha_reciente'  : 'Racha reciente (último año)',
            'goles_contra_avg': 'Goles concedidos (promedio)',
            'diferencia_goles': 'Diferencia de goles',
            'goles_favor_avg' : 'Goles anotados (promedio)',
            'gano_penales_pct': '% Victoria en penales',
            'confederacion_cod': 'Confederación',
            'es_local'        : 'Equipo local (sede)',
        }
        fi['nombre'] = fi['variable'].map(nombres_legibles).fillna(fi['variable'])

        col_fi1, col_fi2 = st.columns([6, 4], gap="large")

        with col_fi1:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            fig3.patch.set_facecolor(BG_DARK)
            ax3.set_facecolor(BG_DARK)

            colores_fi = [GOLD if i == 0 else SILVER if i == 1 else '#CD7F32' if i == 2
                          else '#1F6FEB' for i in range(len(fi))]

            bars3 = ax3.barh(fi['nombre'][::-1], fi['importancia'][::-1] * 100,
                              color=colores_fi[::-1], height=0.65, edgecolor='none')

            for bar, val in zip(bars3, fi['importancia'][::-1] * 100):
                ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f'{val:.1f}%', va='center', ha='left',
                         color=TEXT, fontsize=9, fontweight='bold')

            ax3.set_xlim(0, fi['importancia'].max() * 120)
            ax3.set_xlabel('Importancia (%)', color=SILVER, fontsize=9)
            ax3.tick_params(colors=TEXT, labelsize=9)
            ax3.spines[['top', 'right', 'bottom']].set_visible(False)
            ax3.spines['left'].set_color('#30363D')
            ax3.xaxis.grid(True, color='#30363D', linestyle='--', alpha=0.5)
            ax3.set_axisbelow(True)
            ax3.set_title('Importancia de cada variable (Random Forest)',
                          color=GOLD, fontsize=10, pad=10)

            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        with col_fi2:
            st.markdown("**Detalle por variable**")
            rows_fi = ""
            for i, row in fi.iterrows():
                pct = row['importancia'] * 100
                bar_w = int(pct * 2)
                rows_fi += f"""
                <tr>
                  <td><b>{row['nombre']}</b></td>
                  <td>
                    <div class="bar-wrap">
                      <div class="bar-fill" style="width:{min(bar_w,100)}%;
                           background:linear-gradient(90deg,{GOLD},{GOLD2});">
                        {pct:.1f}%
                      </div>
                    </div>
                  </td>
                </tr>"""

            st.markdown(f"""
            <table class="styled-table">
              <thead><tr><th>Variable</th><th>Importancia</th></tr></thead>
              <tbody>{rows_fi}</tbody>
            </table>""", unsafe_allow_html=True)

            st.info("💡 El **Ranking FIFA** domina con ~42% de importancia, seguido del "
                    "**interés en Google Trends** (~28%) como proxy de popularidad global.")


# ════════════════════════════════════════════
# TAB 4: BACKTESTING
# ════════════════════════════════════════════
with tab4:
    st.subheader("Backtesting Histórico — ¿Cómo hubiera funcionado el modelo?")
    st.caption("Entrenado con datos hasta N-4 años, predice el ganador del siguiente Mundial")

    if data['backtesting'] is None:
        st.warning("Ejecuta `python backtesting.py` para generar backtesting_resultados.csv")
    else:
        bt = data['backtesting'].copy()

        # ── Métricas globales ──
        acc1 = bt['Acierto_Top1'].mean()
        acc3 = bt['Acierto_Top3'].mean()
        acc5 = bt['Acierto_Top5'].mean()
        media_puesto = bt['Puesto_Ganador_Real'].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{acc1:.0%}", "Top-1 Accuracy"),
            (c2, f"{acc3:.0%}", "Top-3 Accuracy"),
            (c3, f"{acc5:.0%}", "Top-5 Accuracy"),
            (c4, f"#{media_puesto:.1f}", "Puesto medio del ganador"),
        ]:
            with col:
                color = GREEN if float(val.replace('%','').replace('#','')) > 50 else GOLD
                st.markdown(f'<div class="metric-card">'
                            f'<div class="metric-val" style="color:{color};">{val}</div>'
                            f'<div class="metric-lbl">{lbl}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabla por mundial ──
        for _, row in bt.iterrows():
            mundial = int(row['Mundial'])
            ganador = row['Ganador_Real']
            pred1   = row['Prediccion_1']
            puesto  = int(row['Puesto_Ganador_Real'])
            prob    = row['Probabilidad_Ganador_%']

            acierto = row['Acierto_Top1']
            acierto3 = row['Acierto_Top3']
            icono   = "✅" if acierto else ("🟡" if acierto3 else "❌")
            color_borde = GREEN if acierto else (GOLD if acierto3 else RED)

            top5_preds = [row.get(f'Prediccion_{i}', '—') for i in range(1, 6)]
            top5_str   = " → ".join([
                f"**{p}** ✅" if p == ganador else p for p in top5_preds
            ])

            st.markdown(f"""
            <div style="background:{BG_CARD}; border-left:4px solid {color_borde};
                        border-radius:8px; padding:1rem 1.4rem; margin-bottom:1rem;">
              <h4 style="color:{GOLD}; margin:0 0 0.5rem 0;">{icono} Mundial {mundial}</h4>
              <table style="width:100%; font-size:0.9rem; color:{TEXT};">
                <tr><td style="width:45%;color:{SILVER};">Ganador real:</td>
                    <td><b style="color:{GOLD};">{ganador}</b></td></tr>
                <tr><td style="color:{SILVER};">Predicción #1 del modelo:</td>
                    <td><b>{pred1}</b></td></tr>
                <tr><td style="color:{SILVER};">Puesto del ganador real:</td>
                    <td><b>#{puesto}</b></td></tr>
                <tr><td style="color:{SILVER};">Prob. asignada al ganador:</td>
                    <td><b>{prob}%</b></td></tr>
              </table>
              <p style="margin:0.6rem 0 0 0; font-size:0.85rem; color:{SILVER};">
                Top 5: {' → '.join(top5_preds)}
              </p>
            </div>""", unsafe_allow_html=True)

        st.info("💡 El modelo no predice partidos individuales, sino probabilidades de ser "
                "campeón basadas en patrones históricos. Puesto #1–3 para el ganador real "
                "es considerado un buen resultado dado el alto azar en torneos eliminatorios.")


# ════════════════════════════════════════════
# TAB 5: COMPARATIVA DE MODELOS
# ════════════════════════════════════════════
with tab5:
    st.subheader("Comparativa de Modelos — AUC-ROC (validación cruzada 5-fold)")
    st.caption("Área bajo la curva ROC: mide la capacidad del modelo para distinguir "
               "campeones de no campeones. Mayor = mejor.")

    comp_df = data['comparativa'] if data['comparativa'] is not None else data['metricas']

    if comp_df is None:
        st.warning("Ejecuta `python modelo_prediccion.py` para generar comparativa_modelos.csv")
    else:
        # Filtrar solo filas con AUC válido
        if 'AUC_ROC_media' in comp_df.columns:
            comp_plot = comp_df[comp_df['AUC_ROC_media'].notna()].copy()
            modelos   = comp_plot['Modelo'].tolist()
            aucs      = comp_plot['AUC_ROC_media'].tolist()
            stds      = comp_plot.get('AUC_ROC_std', pd.Series([0]*len(aucs))).tolist()
        else:
            # Parsear desde columna texto "0.9660 ± 0.0509"
            def parse_auc(s):
                try:
                    parts = str(s).split('±')
                    return float(parts[0].strip()), float(parts[1].strip()) if len(parts) > 1 else 0
                except:
                    return None, None

            comp_df[['auc_val', 'auc_std']] = comp_df.iloc[:, 1].apply(
                lambda x: pd.Series(parse_auc(x))
            )
            comp_plot = comp_df[comp_df['auc_val'].notna()].copy()
            modelos   = comp_plot.iloc[:, 0].tolist()
            aucs      = comp_plot['auc_val'].tolist()
            stds      = comp_plot['auc_std'].tolist()

        col_g, col_t = st.columns([6, 4], gap="large")

        with col_g:
            fig4, ax4 = plt.subplots(figsize=(8, 4.5))
            fig4.patch.set_facecolor(BG_DARK)
            ax4.set_facecolor(BG_DARK)

            colors_m = [GOLD, '#2EA043', '#1F6FEB', SILVER][:len(modelos)]
            x_pos = range(len(modelos))

            bars4 = ax4.bar(x_pos, aucs, color=colors_m, width=0.55,
                             edgecolor='none',
                             yerr=stds if any(s > 0 for s in stds) else None,
                             capsize=5, error_kw={'color': TEXT, 'linewidth': 1.5})

            for bar, val, std in zip(bars4, aucs, stds):
                label = f'{val:.4f}\n±{std:.4f}' if std > 0 else f'{val:.4f}'
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         label, ha='center', va='bottom',
                         color=TEXT, fontsize=8.5, fontweight='bold')

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(modelos, color=TEXT, fontsize=9)
            ax4.set_ylim(0.88, 1.01)
            ax4.set_ylabel('AUC-ROC', color=SILVER, fontsize=9)
            ax4.tick_params(colors=SILVER, labelsize=8.5)
            ax4.spines[['top', 'right']].set_visible(False)
            ax4.spines[['left', 'bottom']].set_color('#30363D')
            ax4.yaxis.grid(True, color='#30363D', linestyle='--', alpha=0.5)
            ax4.set_axisbelow(True)
            ax4.axhline(y=0.9, color=GOLD, linestyle=':', alpha=0.5, linewidth=1)
            ax4.text(len(modelos) - 0.4, 0.902, 'Umbral 0.90', color=GOLD,
                     fontsize=7.5, alpha=0.8)
            ax4.set_title('AUC-ROC por modelo (5-fold cross-validation)',
                          color=GOLD, fontsize=10, pad=10)

            fig4.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        with col_t:
            st.markdown("**Métricas detalladas**")
            rows_m = ""
            for modelo, auc, std in zip(modelos, aucs, stds):
                best = auc == max(aucs)
                color_modelo = GOLD if best else TEXT
                rows_m += f"""
                <tr>
                  <td style="color:{color_modelo};"><b>{'🏆 ' if best else ''}{modelo}</b></td>
                  <td style="text-align:center;"><b>{auc:.4f}</b></td>
                  <td style="text-align:center;color:{SILVER};">±{std:.4f}</td>
                </tr>"""

            st.markdown(f"""
            <table class="styled-table">
              <thead><tr>
                <th>Modelo</th><th style="text-align:center;">AUC-ROC</th>
                <th style="text-align:center;">Desv. Est.</th>
              </tr></thead>
              <tbody>{rows_m}</tbody>
            </table>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:{BG_MED}; border-radius:8px; padding:1rem;
                        font-size:0.85rem; color:{SILVER};">
            <b style="color:{GOLD};">¿Qué es AUC-ROC?</b><br><br>
            Mide la probabilidad de que el modelo rankee un campeón histórico por
            encima de un no campeón.<br><br>
            • <b>1.00</b> = perfecto<br>
            • <b>0.90+</b> = excelente<br>
            • <b>0.80-0.89</b> = bueno<br>
            • <b>0.50</b> = aleatorio
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="gold">', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:{SILVER}; font-size:0.8rem; padding:0.5rem;'>
  ⚽ <b style="color:{GOLD};">dataGol</b> — Predictor del Mundial 2026 &nbsp;|&nbsp;
  Machine Learning + Monte Carlo + Análisis Estadístico &nbsp;|&nbsp;
  Datos: Kaggle · FIFA API · Transfermarkt
</div>
""", unsafe_allow_html=True)
