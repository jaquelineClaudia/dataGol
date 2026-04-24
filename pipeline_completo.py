"""
pipeline_completo.py
===================
Ejecuta el pipeline completo de dataGol en orden y detiene si falla algun paso.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Preparacion de datos", "preparar_datos.py"),
    ("Entrenamiento de modelos", "modelo_prediccion.py"),
    ("Analisis estadistico", "analisis_estadistico.py"),
    ("Backtesting", "backtesting.py"),
    ("Simulacion Monte Carlo", "simulacion_mundial_2026.py"),
    ("Simulacion bracket", "simulacion_bracket.py"),
    ("Visualizaciones", "visualizaciones.py"),
]


def run_step(title: str, script: str) -> None:
    print("=" * 68)
    print(f"  {title}: {script}")
    print("=" * 68)

    cmd = [sys.executable, str(BASE_DIR / script)]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Fallo en {script} (exit code {result.returncode})")


if __name__ == "__main__":
    print("\nIniciando pipeline completo dataGol...\n")

    for title, script in STEPS:
        run_step(title, script)

    print("\nPipeline completado correctamente.")
    print("Archivos clave generados:")
    print("- dataset_modelo.csv")
    print("- probabilidades_ganador_2026.csv")
    print("- backtesting_resultados.csv")
    print("- simulacion_monte_carlo_2026.csv")
    print("- bracket_simulado.csv")
    print("- probabilidades_por_fase.csv")
