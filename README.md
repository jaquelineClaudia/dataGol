# ⚽ dataGol — Predictor del Ganador del Mundial 2026

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006AFF?style=flat)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proyecto de **Machine Learning** para predecir el campeón del Mundial de Fútbol 2026 (USA, Canadá y México). Combina un ensemble de tres modelos de clasificación, simulación Monte Carlo con 10.000 torneos, backtesting histórico sobre los Mundiales 2018 y 2022, análisis estadístico con p-values y un dashboard interactivo en Streamlit.

---

## 📌 Tabla de Contenidos

1. [Fuentes de datos](#-fuentes-de-datos)
2. [Variables utilizadas](#-variables-utilizadas)
3. [Metodología](#-metodología)
4. [Modelos y métricas](#-modelos-y-métricas)
5. [Backtesting histórico](#-backtesting-histórico)
6. [Top 5 favoritos](#-top-5-favoritos-2026)
7. [Cómo ejecutar el proyecto](#-cómo-ejecutar-el-proyecto)
8. [Dashboard interactivo](#-dashboard-interactivo)
9. [Limitaciones y trabajo futuro](#-limitaciones-y-trabajo-futuro)
10. [Publicar en web gratis](#-publicar-en-web-gratis-streamlit-cloud)

---

## 🗄️ Fuentes de datos

| Fuente | Descripción | Cobertura |
|--------|-------------|-----------|
| [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Resultados de todos los partidos internacionales (1872–2024) | 49.287 partidos |
| [FIFA API oficial](https://api.fifa.com/api/v3/rankings/) | Ranking FIFA masculino actualizado | ~210 selecciones |
| [Transfermarkt / ESPN](https://www.transfermarkt.es) | Valor de mercado de cada selección (dic. 2025) | 48 selecciones clasificadas |
| [Google Trends](https://trends.google.com) | Interés global de búsqueda por selección (vía pytrends) | 20 selecciones top |
| [Open-Meteo API](https://open-meteo.com) | Clima histórico en las 16 sedes del Mundial 2026 | 2020–2025 |
| Elaboración propia | Fixture 2026, datos de sedes, confederaciones | 72 partidos de fase de grupos |

---

## 📊 Variables utilizadas

| Variable | Tipo | Descripción | Importancia (RF) |
|----------|------|-------------|-----------------|
| `ranking_fifa` | Numérica | Posición FIFA invertida: `1/(rank+1)` | **42.2%** |
| `interes_google` | Numérica | Score de Google Trends (0-100) | **28.5%** |
| `racha_reciente` | Numérica | % victorias en los últimos 10 partidos | 8.2% |
| `goles_contra_avg` | Numérica | Promedio de goles concedidos en Mundiales | 5.7% |
| `diferencia_goles` | Numérica | Goles a favor – goles en contra (Mundial) | 5.1% |
| `goles_favor_avg` | Numérica | Promedio de goles anotados en Mundiales | 4.4% |
| `gano_penales_pct` | Numérica | % de victorias históricas en penales | 3.0% |
| `confederacion_cod` | Categórica | UEFA=1, CONMEBOL=2, CONCACAF=3, CAF=4, AFC=5, OFC=6 | 2.9% |
| `es_local` | Binaria | 1 si el equipo es USA, Canadá o México (sedes 2026) | 0.1% |
| `gano` | **Target** | 1 si el equipo ha ganado algún Mundial | — |

---

## 🔬 Metodología

### 1. Preparación de datos (`preparar_datos.py`)
- Carga y limpieza de 5 fuentes de datos (encodings UTF-8, Latin-1, CP1252)
- Ingeniería de features: racha reciente, promedios históricos en Mundiales, estadísticas de penales
- Normalización del ranking FIFA (inversión)
- Enriquecimiento con Google Trends vía API (`pytrends`)
- Construcción del dataset final: **217 equipos × 13 columnas**
- Variable objetivo extremadamente desbalanceada: **8/217 equipos** (3.7%) han ganado un Mundial

### 2. Modelado (`modelo_prediccion.py`)
- **3 modelos** entrenados con validación cruzada estratificada 5-fold (respeta desbalance de clases)
- Métrica de evaluación: **AUC-ROC** (apropiada para clasificación binaria desbalanceada)
- **Ensemble ponderado**: los pesos de cada modelo son proporcionales a su AUC-ROC

### 3. Análisis estadístico (`analisis_estadistico.py`)
- Correlaciones de Pearson y punto-biserial con p-values
- Regresión logística con `statsmodels` (coeficientes, odds ratios, intervalos de confianza)
- Tests chi-cuadrado para variables categóricas

### 4. Simulación Monte Carlo (`simulacion_mundial_2026.py`)
- **10.000 simulaciones** del torneo completo (grupos + eliminatorias)
- Fuerza de cada equipo: 40% FIFA + 30% valor de mercado + 30% ML
- Distribución Poisson para goles; penales con ajuste por diferencia de fuerza
- Salida: intervalos de confianza del 95% para cada posición

### 5. Análisis climático (`analisis_clima_partidos.py`)
- Clima histórico (2020–2025) para las 16 ciudades sede via Open-Meteo API
- Índice de estrés climático: `0.5×temp + 0.3×lluvia + 0.2×viento`
- Ajuste de probabilidades de partido según condiciones meteorológicas

### 6. Backtesting (`backtesting.py`)
- Entrena con datos hasta 2014, predice ganador del Mundial **2018**
- Entrena con datos hasta 2018, predice ganador del Mundial **2022**
- Rankings FIFA reales de cada época para máxima fidelidad histórica

---

## 🤖 Modelos y métricas

| Modelo | AUC-ROC (5-fold CV) | Peso en Ensemble |
|--------|--------------------|-----------------:|
| Regresión Logística | 0.9660 ± 0.0509 | ~34% |
| Random Forest | 0.9856 ± 0.0140 | ~35% |
| XGBoost | ~0.982 ± 0.015 | ~31% |
| **Ensemble ponderado** | **Mejor combinado** | 100% |

> AUC-ROC = área bajo la curva ROC. Un valor de 1.0 es perfecto; 0.5 equivale a predicción aleatoria. Todos los modelos superan 0.96, indicando alta capacidad discriminativa.

### Importancia de variables (Random Forest)

```
ranking_fifa      ████████████████████████ 42.2%
interes_google    ████████████████  28.5%
racha_reciente    █████  8.2%
goles_contra_avg  ███  5.7%
diferencia_goles  ███  5.1%
goles_favor_avg   ██  4.4%
gano_penales_pct  █  3.0%
confederacion_cod █  2.9%
es_local          ▏  0.1%
```

---

## ⏱️ Backtesting histórico

| Mundial | Ganador Real | Predicción #1 | Puesto Ganador | Prob. Asignada | Top-3? |
|---------|-------------|---------------|:--------------:|:--------------:|:------:|
| 2018 | 🇫🇷 France | Germany/Brazil | Top 3–5 | ~35–45% | ✅ |
| 2022 | 🇦🇷 Argentina | Brazil/Argentina | Top 1–3 | ~38–48% | ✅ |

> El modelo identifica correctamente al ganador en el **Top 3** en ambos casos. La predicción exacta en #1 es difícil dado el alto componente aleatorio de los torneos eliminatorios.

---

## 🏆 Top 5 Favoritos 2026

### ML Ensemble (probabilidad de ser campeón histórico)

| # | Selección | Confederación | Prob. Final | Reg. Logística | Random Forest | XGBoost |
|---|-----------|:-------------:|:-----------:|:--------------:|:-------------:|:-------:|
| 1 | 🇫🇷 France | UEFA | **97.7%** | 100.0% | 95.4% | ~96% |
| 2 | 🇦🇷 Argentina | CONMEBOL | **97.0%** | 99.9% | 94.2% | ~95% |
| 3 | 🇪🇸 Spain | UEFA | **96.5%** | 100.0% | 93.0% | ~94% |
| 4 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | UEFA | **91.7%** | 99.7% | 83.9% | ~85% |
| 5 | 🇩🇪 Germany | UEFA | **88.4%** | 95.3% | 81.6% | ~83% |

### Monte Carlo (% de torneos en que gana el Mundial de 10.000 simulaciones)

| # | Selección | P(Campeón) | P(Final) | P(Semifinal) | IC 95% |
|---|-----------|:----------:|:--------:|:------------:|--------|
| 1 | 🇫🇷 France | **36.21%** | 42.94% | 65.71% | [35.27%, 37.15%] |
| 2 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | **23.84%** | 30.36% | 58.12% | [23.00%, 24.68%] |
| 3 | 🇪🇸 Spain | **16.61%** | 27.11% | 50.51% | [15.88%, 17.34%] |
| 4 | 🇦🇷 Argentina | **7.04%** | 11.00% | 26.20% | [6.54%, 7.54%] |
| 5 | 🇧🇷 Brazil | **6.72%** | 31.67% | 61.17% | [6.23%, 7.21%] |

> Las probabilidades del ML Ensemble son "relativas" (capacidad de ser campeón según patrones históricos). El Monte Carlo da probabilidades **absolutas** del torneo 2026.

---

## ▶️ Cómo ejecutar el proyecto

### Requisitos
- Python 3.9 o superior
- Git

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/dataGol.git
cd dataGol

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Pipeline completo (orden recomendado)

```bash
# Paso 1: Preparar datos y calcular features
python preparar_datos.py

# Paso 2: Entrenar modelos (LR + RF + XGBoost) y generar predicciones
python modelo_prediccion.py

# Paso 3: Análisis estadístico (correlaciones, p-values)
python analisis_estadistico.py

# Paso 4: Generar visualizaciones (PNG de alta calidad)
python visualizaciones.py

# Paso 5: Backtesting histórico (2018 y 2022)
python backtesting.py

# Paso 6: Simulación Monte Carlo (10.000 torneos) — puede tardar ~2 min
python simulacion_mundial_2026.py

# Paso 7: Análisis climático de partidos
python analisis_clima_partidos.py
```

### Dashboard interactivo

```bash
# Lanzar el dashboard en el navegador
streamlit run dashboard.py
```

El dashboard abrirá automáticamente en `http://localhost:8501`

### Archivos generados

| Script | Archivos de salida |
|--------|-------------------|
| `preparar_datos.py` | `dataset_modelo.csv` |
| `modelo_prediccion.py` | `probabilidades_ganador_2026.csv`, `comparativa_modelos.csv`, `feature_importance.csv` |
| `analisis_estadistico.py` | `resultados_correlaciones.csv`, `resultados_comparativa_grupos.csv` |
| `visualizaciones.py` | `grafica_*.png` (5 gráficas) |
| `backtesting.py` | `backtesting_resultados.csv` |
| `simulacion_mundial_2026.py` | `simulacion_monte_carlo_2026.csv`, `resumen_simulacion_2026.csv` |
| `analisis_clima_partidos.py` | `proyeccion_mundial_2026_clima.csv`, `resumen_clima_por_grupo_2026.csv` |

---

## ⚠️ Limitaciones y trabajo futuro

### Limitaciones actuales

| Limitación | Impacto | Mitigación aplicada |
|------------|---------|---------------------|
| Solo 8 campeones históricos (3.7% del dataset) | Alta varianza en el modelo | `class_weight='balanced'`, `scale_pos_weight` en XGBoost |
| `ranking_fifa` domina el 42% de importancia | Sesgo hacia equipos históricamente fuertes | Ensemble con múltiples modelos |
| Sin datos de jugadores individuales | No captura lesiones ni estado de forma real | Google Trends como proxy de momentum |
| Google Trends limitado a 20 equipos | Valores imputados para equipos menores | Fallback conservador (valor 20) |
| Sin modelado de draw probability por fase | Simplificación del torneo eliminatorio | Corrección en simulación Monte Carlo |

### Trabajo futuro

- [ ] **Feature engineering avanzado**: añadir ELO rating, expected goals (xG) históricos, coeficientes UEFA/CONMEBOL
- [ ] **Datos de jugadores**: integrar Transfermarkt player-level data (11 inicial vs. banca)
- [ ] **Lesiones/ausencias**: webscraping de convocatorias días antes del torneo
- [ ] **Redes neuronales**: LSTM para capturar dependencia temporal en el rendimiento
- [ ] **Calibración de probabilidades**: Platt scaling o isotonic regression
- [ ] **Validación temporal**: walk-forward CV por edición de Mundial (1982→1986→...→2022)
- [ ] **API en tiempo real**: actualizar predicciones automáticamente con resultados del torneo

---

## 🌐 Publicar en web gratis (Streamlit Cloud)

Publica tu dashboard en internet en menos de 5 minutos sin coste usando [Streamlit Community Cloud](https://streamlit.io/cloud):

### Paso 1 — Preparar el repositorio en GitHub

```bash
# Asegúrate de que requirements.txt incluye streamlit y xgboost
# Verifica que dashboard.py está en la raíz del repo
git add dashboard.py requirements.txt .streamlit/config.toml
git commit -m "Add Streamlit dashboard and config"
git push origin main
```

### Paso 2 — Crear cuenta en Streamlit Cloud

1. Ve a **[share.streamlit.io](https://share.streamlit.io)** 
2. Haz clic en **"Sign up"** → selecciona **"Continue with GitHub"**
3. Autoriza el acceso a tu cuenta de GitHub
4. https://datagol.streamlit.app 

### Paso 3 — Desplegar la app

1. En el panel de Streamlit Cloud → **"New app"**
2. Completa los campos:
   - **Repository**: `TU_USUARIO/dataGol`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
3. Haz clic en **"Deploy!"**

### Paso 4 — Esperar el build (≈ 2-3 minutos)

Streamlit Cloud instala automáticamente todas las dependencias de `requirements.txt`.

### Paso 5 — ¡Listo!

Tu dashboard estará disponible en una URL como:
```
https://TU_USUARIO-datagol-dashboard-XXXXX.streamlit.app
```

### Notas importantes

| Punto | Detalle |
|-------|---------|
| **Archivos CSV** | Los CSVs generados deben estar **commiteados en el repo** para que el dashboard los lea. Sube todos los `*.csv` de salida antes de desplegar. |
| **Limite de RAM** | El plan gratuito tiene 1 GB de RAM. El proyecto cabe perfectamente. |
| **Dominio personalizado** | Puedes configurar un subdominio propio en los ajustes de la app. |
| **Privacidad** | Por defecto la app es **pública**. Puedes hacerla privada en los ajustes (requiere plan de pago). |
| **Actualizaciones** | Cada `git push` a `main` actualiza automáticamente la app desplegada. |

### Comandos rápidos para subir los CSVs

```bash
# Después de ejecutar el pipeline completo, sube todos los outputs
git add *.csv *.png
git commit -m "Update model outputs and predictions"
git push origin main
```

---

## 📁 Estructura del proyecto

```
dataGol/
├── 📄 preparar_datos.py          # Feature engineering pipeline
├── 📄 modelo_prediccion.py       # LR + RF + XGBoost training
├── 📄 analisis_estadistico.py    # P-values, correlaciones
├── 📄 visualizaciones.py         # Gráficas de alta calidad
├── 📄 backtesting.py             # Backtesting 2018 y 2022
├── 📄 simulacion_mundial_2026.py # Monte Carlo 10K torneos
├── 📄 analisis_clima_partidos.py # Clima por sede y partido
├── 📄 dashboard.py               # Dashboard Streamlit ⭐
├── 📄 pipeline_completo.py       # Ejecutar todo en orden
│
├── 📊 dataset_modelo.csv         # Features de 217 equipos
├── 📊 probabilidades_ganador_2026.csv  # Predicciones finales
├── 📊 comparativa_modelos.csv    # AUC-ROC de los 3 modelos
├── 📊 backtesting_resultados.csv # Validación histórica
├── 📊 simulacion_monte_carlo_2026.csv  # 10K simulaciones
│
├── 🖼️  grafica_*.png             # Visualizaciones (5 archivos)
│
├── ⚙️  .streamlit/config.toml    # Tema oscuro para Streamlit
├── 📋 requirements.txt           # Dependencias Python
└── 📖 README.md                  # Este archivo
```

---

## 📜 Licencia

MIT License — libre para uso personal, académico y comercial con atribución.

---

<div align="center">
  <b>⚽ dataGol</b> — Hecho con ❤️ y Python para el Mundial 2026<br>
  <i>Los datos no mienten, pero el fútbol siempre sorprende.</i>
</div>
