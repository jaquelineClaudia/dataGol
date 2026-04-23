# 🏆 dataGol — Instrucciones para Claude Code

Este archivo le dice a Claude Code exactamente qué hacer con este repositorio.

## Objetivo
Construir un modelo predictivo del ganador del Mundial 2026 usando los datos del repo,
añadiendo nuevas variables, análisis estadístico completo (p-values, regresión logística,
feature importance) y visualizaciones.

## Archivos que debe crear o completar
- `preparar_datos.py` → limpia y une todos los CSVs, calcula variables nuevas
- `modelo_prediccion.py` → entrena regresión logística + Random Forest
- `analisis_estadistico.py` → p-values, correlaciones, significancia
- `visualizaciones.py` → gráficas de resultados
- `pipeline_completo.py` → corre todo en orden
- `requirements.txt` → dependencias

## Orden de ejecución
```
pip install -r requirements.txt
python preparar_datos.py
python modelo_prediccion.py
python analisis_estadistico.py
python visualizaciones.py
```
O simplemente:
```
python pipeline_completo.py
```
