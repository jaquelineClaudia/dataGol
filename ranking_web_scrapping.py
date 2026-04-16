import csv
from pathlib import Path

import requests


# Endpoint oficial de FIFA para obtener el ranking mundial.
FIFA_RANKING_API = "https://api.fifa.com/api/v3/rankings/"
# Nombre del archivo CSV de salida que se creara en esta misma carpeta.
OUTPUT_FILE = "ranking_fifa_masculino.csv"


def extract_team_name(team_name_entries):
    # TeamName suele venir como lista de objetos, por ejemplo:
    # [{"Locale": "en-GB", "Description": "France"}]
    # Esta funcion extrae solo el texto del nombre del pais/equipo.
    if not isinstance(team_name_entries, list) or not team_name_entries:
        return ""
    first = team_name_entries[0]
    if isinstance(first, dict):
        return str(first.get("Description", "")).strip()
    return str(first).strip()


def fetch_mens_ranking(count=500):
    # Cabeceras para simular una peticion de navegador y recibir JSON.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    }

    # gender=1 corresponde al ranking masculino.
    # count define cuantos registros maximo queremos pedir.
    params = {"gender": 1, "count": count}

    # Llamada HTTP a la API de FIFA.
    response = requests.get(FIFA_RANKING_API, headers=headers, params=params, timeout=30)
    # Si hay error HTTP (404, 500, etc.), lanza excepcion para detectarlo rapido.
    response.raise_for_status()
    # Convertimos la respuesta JSON en un diccionario de Python.
    payload = response.json()

    # Transformamos cada fila de la API a un formato mas limpio para el CSV.
    rows = []
    for item in payload.get("Results", []):
        rows.append(
            {
                "rank": item.get("Rank"),
                "pais": extract_team_name(item.get("TeamName")),
                "codigo_pais": item.get("IdCountry"),
                "confederacion": item.get("ConfederationName"),
                "puntos": item.get("DecimalTotalPoints"),
                "puntos_previos": item.get("DecimalPrevPoints"),
                "posicion_previa": item.get("PrevRank"),
                "movimiento": item.get("RankingMovement"),
                "partidos": item.get("Matches"),
                "fecha_publicacion": item.get("PubDate"),
                "proxima_publicacion": item.get("NextPubDate"),
            }
        )

    # Orden final por posicion para asegurar salida de 1...N.
    rows.sort(key=lambda row: (row["rank"] is None, row["rank"]))
    return rows


def save_to_csv(rows, output_path):
    # Orden de columnas del archivo CSV.
    fieldnames = [
        "rank",
        "pais",
        "codigo_pais",
        "confederacion",
        "puntos",
        "puntos_previos",
        "posicion_previa",
        "movimiento",
        "partidos",
        "fecha_publicacion",
        "proxima_publicacion",
    ]

    # Escritura del CSV con codificacion UTF-8 para soportar acentos.
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Primera fila: nombres de columnas.
        writer.writeheader()
        # Filas de datos del ranking.
        writer.writerows(rows)


def main():
    # 1) Obtener ranking completo desde FIFA.
    rows = fetch_mens_ranking()

    # 2) Construir ruta absoluta del archivo CSV dentro de esta carpeta.
    output_path = Path(__file__).resolve().parent / OUTPUT_FILE

    # 3) Guardar los datos en CSV.
    save_to_csv(rows, output_path)

    # 4) Mostrar resumen por consola.
    print(f"CSV generado: {output_path}")
    print(f"Equipos exportados: {len(rows)}")


# Este bloque permite ejecutar el script directamente desde terminal.
if __name__ == "__main__":
    main()
