import argparse
from dataclasses import dataclass
import time
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FALLBACK_COORDS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("Lusail", "Qatar"): (25.4202, 51.4903),
    ("Doha", "Qatar"): (25.2854, 51.5310),
    ("Al Rayyan", "Qatar"): (25.2919, 51.4244),
    ("Al Khor", "Qatar"): (25.6839, 51.5058),
    ("New York/New Jersey", "United States"): (40.8136, -74.0745),
    ("San Francisco Bay Area", "United States"): (37.4021, -121.9780),
}

HOST_COUNTRY_BY_CITY = {
    "Mexico City": "Mexico",
    "Guadalajara": "Mexico",
    "Monterrey": "Mexico",
    "Toronto": "Canada",
    "Vancouver": "Canada",
}

TEAM_ALIASES = {
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "Iran": "IR Iran",
    "Turkey": "Turkiye",
    "Turkiye": "Turkiye",
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "Cape Verde": "Cabo Verde",
}


@dataclass
class Config:
    fixtures_path: str
    ranking_path: str
    out_csv_path: str
    group_summary_csv_path: str
    high_risk_csv_path: str
    climate_history_start: str
    climate_history_end: str


def normalize_team_name(name: str) -> str:
    fixed = TEAM_ALIASES.get(name, name)
    fixed = unicodedata.normalize("NFKD", str(fixed)).encode("ascii", "ignore").decode("ascii")
    return TEAM_ALIASES.get(fixed.strip(), fixed.strip())


def infer_country(city: str) -> str:
    return HOST_COUNTRY_BY_CITY.get(city, "United States")


def normalize_city_name(city: str) -> str:
    value = str(city).strip()
    if "(" in value:
        value = value.split("(", 1)[0].strip()
    return value


def load_fixtures(cfg: Config) -> pd.DataFrame:
    fixtures = pd.read_csv(cfg.fixtures_path)

    fixtures = fixtures.rename(
        columns={
            "Fecha": "date",
            "Equipo_1": "home_team",
            "Equipo_2": "away_team",
            "Sede": "city",
        }
    )

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.dropna(subset=["date", "home_team", "away_team", "city"])

    fixtures["city"] = fixtures["city"].map(normalize_city_name)
    fixtures["country"] = fixtures["city"].map(infer_country)
    fixtures["match_date"] = fixtures["date"].dt.date.astype(str)
    fixtures["month_day"] = fixtures["date"].dt.strftime("%m-%d")

    fixtures["home_team_rank_name"] = fixtures["home_team"].map(normalize_team_name)
    fixtures["away_team_rank_name"] = fixtures["away_team"].map(normalize_team_name)

    return fixtures.sort_values("date").reset_index(drop=True)


def load_ranking(cfg: Config) -> pd.DataFrame:
    ranking = pd.read_csv(cfg.ranking_path)
    ranking["pais_norm"] = ranking["pais"].astype(str).map(normalize_team_name)
    return ranking[["pais_norm", "puntos", "rank"]].drop_duplicates(subset=["pais_norm"])


def geocode_city(city: str, country: str) -> Tuple[float, float] | None:
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    results = payload.get("results") or []
    if not results:
        return None

    by_country = [r for r in results if str(r.get("country", "")).lower() == country.lower()]
    chosen = by_country[0] if by_country else results[0]
    return float(chosen["latitude"]), float(chosen["longitude"])


def build_city_coordinates(fixtures: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    pairs = (
        fixtures[["city", "country"]]
        .drop_duplicates()
        .sort_values(["country", "city"])
        .itertuples(index=False, name=None)
    )

    coords: Dict[Tuple[str, str], Tuple[float, float]] = {}
    missing: List[Tuple[str, str]] = []

    for city, country in pairs:
        fallback = FALLBACK_COORDS.get((city, country))
        if fallback is not None:
            coords[(city, country)] = fallback
            continue

        try:
            point = geocode_city(city, country)
            if point is None:
                missing.append((city, country))
            else:
                coords[(city, country)] = point
        except requests.RequestException:
            missing.append((city, country))

    if missing:
        print("Ciudades sin coordenadas (usar fallback manual):")
        for city, country in missing:
            print(f"  - {city}, {country}")

    return coords


def fetch_city_daily_weather(city: str, country: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "UTC",
    }

    payload = None
    for attempt in range(4):
        response = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if response.status_code == 429:
            # Reintento con espera creciente para respetar limite de tasa.
            time.sleep(2 + attempt * 3)
            continue
        response.raise_for_status()
        payload = response.json()
        break

    if payload is None:
        raise requests.HTTPError(f"429 Too Many Requests para {city}, {country}")

    daily = payload.get("daily", {})

    df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temp_mean": daily.get("temperature_2m_mean", []),
            "precip_sum": daily.get("precipitation_sum", []),
            "wind_max": daily.get("wind_speed_10m_max", []),
        }
    )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]) 
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df["city"] = city
    df["country"] = country
    return df


def build_climate_normals(fixtures: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    coords = build_city_coordinates(fixtures)
    chunks = []
    failed_cities: List[Tuple[str, str, float, float]] = []

    for (city, country), (lat, lon) in coords.items():
        try:
            city_weather = fetch_city_daily_weather(
                city,
                country,
                lat,
                lon,
                cfg.climate_history_start,
                cfg.climate_history_end,
            )
            if not city_weather.empty:
                chunks.append(city_weather)
        except requests.RequestException as exc:
            print(f"No se pudo descargar clima para {city}, {country}: {exc}")
            failed_cities.append((city, country, lat, lon))

    # Segundo intento solo para las sedes fallidas usando una ventana mas corta.
    if failed_cities:
        print("\nReintentando sedes fallidas con ventana climatica reducida...")
        short_start = max(cfg.climate_history_start, "2023-01-01")
        for city, country, lat, lon in failed_cities:
            try:
                city_weather = fetch_city_daily_weather(
                    city,
                    country,
                    lat,
                    lon,
                    short_start,
                    cfg.climate_history_end,
                )
                if not city_weather.empty:
                    chunks.append(city_weather)
                    print(f"  - Recuperado: {city}, {country}")
            except requests.RequestException as exc:
                print(f"  - Sigue fallando: {city}, {country} ({exc})")
            time.sleep(2)

    if not chunks:
        return pd.DataFrame(columns=["city", "country", "month_day", "temp_mean", "precip_sum", "wind_max"])

    weather = pd.concat(chunks, ignore_index=True)
    normals = (
        weather.groupby(["city", "country", "month_day"], as_index=False)
        .agg(
            temp_mean=("temp_mean", "mean"),
            precip_sum=("precip_sum", "mean"),
            wind_max=("wind_max", "mean"),
        )
    )

    return normals


def add_weather_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rain_bin"] = np.where(out["precip_sum"].fillna(0) > 1.0, "lluvia", "sin_lluvia")

    out["temp_bin"] = pd.cut(
        out["temp_mean"],
        bins=[-np.inf, 15, 22, 30, np.inf],
        labels=["frio", "templado", "calido", "muy_calido"],
    ).astype("object")
    out["temp_bin"] = out["temp_bin"].fillna("desconocido")

    out["wind_bin"] = pd.cut(
        out["wind_max"],
        bins=[-np.inf, 15, 25, np.inf],
        labels=["viento_bajo", "viento_medio", "viento_alto"],
    ).astype("object")
    out["wind_bin"] = out["wind_bin"].fillna("desconocido")

    return out


def sigmoid(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))


def climate_stress_index(df: pd.DataFrame) -> pd.Series:
    temp_stress = np.clip((df["temp_mean"].fillna(20) - 22) / 12, 0, 1)
    rain_stress = np.clip(df["precip_sum"].fillna(0) / 8, 0, 1)
    wind_stress = np.clip((df["wind_max"].fillna(10) - 18) / 18, 0, 1)
    return 0.5 * temp_stress + 0.3 * rain_stress + 0.2 * wind_stress


def climate_risk_label(stress: pd.Series) -> pd.Series:
    return pd.cut(
        stress,
        bins=[-np.inf, 0.25, 0.45, np.inf],
        labels=["bajo", "medio", "alto"],
    ).astype("object").fillna("medio")


def build_2026_projection(fixtures: pd.DataFrame, ranking: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    out = fixtures.merge(
        ranking.rename(columns={"pais_norm": "home_team_rank_name", "puntos": "home_points", "rank": "home_rank"}),
        on="home_team_rank_name",
        how="left",
    )

    out = out.merge(
        ranking.rename(columns={"pais_norm": "away_team_rank_name", "puntos": "away_points", "rank": "away_rank"}),
        on="away_team_rank_name",
        how="left",
    )

    out = out.merge(normals, on=["city", "country", "month_day"], how="left")
    out = add_weather_bins(out)

    out["home_points"] = out["home_points"].fillna(ranking["puntos"].median())
    out["away_points"] = out["away_points"].fillna(ranking["puntos"].median())

    out["strength_diff"] = out["home_points"] - out["away_points"]
    out["base_home_win_prob"] = sigmoid(out["strength_diff"] / 120)

    out["climate_stress"] = climate_stress_index(out)
    out["climate_risk"] = climate_risk_label(out["climate_stress"])

    # Con mas estres climatico, sube empate y baja diferencia esperada.
    out["draw_prob"] = np.clip(0.22 + 0.18 * out["climate_stress"], 0.15, 0.45)

    decisive_mass = 1 - out["draw_prob"]
    out["home_win_prob"] = decisive_mass * out["base_home_win_prob"]
    out["away_win_prob"] = decisive_mass * (1 - out["base_home_win_prob"])

    out["base_goal_diff"] = out["strength_diff"] / 180
    out["climate_adjusted_goal_diff"] = out["base_goal_diff"] * (1 - 0.4 * out["climate_stress"])

    out["expected_total_goals"] = np.clip(2.7 - 0.8 * out["climate_stress"], 1.4, 3.3)

    return out


def print_projection_summary(df: pd.DataFrame) -> None:
    print("\n===== Proyeccion Mundial 2026 (ajustada por clima) =====")
    print(f"Partidos: {len(df)}")
    coverage = df["temp_mean"].notna().mean() * 100
    print(f"Cobertura de clima: {coverage:.1f}%")

    print("\nTop 12 partidos con mayor estres climatico:")
    cols = [
        "date",
        "home_team",
        "away_team",
        "city",
        "temp_mean",
        "precip_sum",
        "wind_max",
        "climate_stress",
        "expected_total_goals",
    ]
    print(
        df.sort_values("climate_stress", ascending=False)[cols]
        .head(12)
        .to_string(index=False)
    )

    print("\nTop 12 favoritos (probabilidad de ganar):")
    fav = df.copy()
    fav["favorite_team"] = np.where(fav["home_win_prob"] >= fav["away_win_prob"], fav["home_team"], fav["away_team"])
    fav["favorite_prob"] = np.where(fav["home_win_prob"] >= fav["away_win_prob"], fav["home_win_prob"], fav["away_win_prob"])
    print(
        fav[["date", "home_team", "away_team", "city", "favorite_team", "favorite_prob", "draw_prob"]]
        .sort_values("favorite_prob", ascending=False)
        .head(12)
        .to_string(index=False)
    )


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Grupo", as_index=False)
        .agg(
            partidos=("Grupo", "count"),
            temp_promedio=("temp_mean", "mean"),
            lluvia_promedio=("precip_sum", "mean"),
            viento_promedio=("wind_max", "mean"),
            stress_promedio=("climate_stress", "mean"),
            prob_empate_promedio=("draw_prob", "mean"),
            goles_esperados_promedio=("expected_total_goals", "mean"),
        )
        .sort_values("stress_promedio", ascending=False)
        .reset_index(drop=True)
    )

    summary["riesgo_climatico_grupo"] = pd.cut(
        summary["stress_promedio"],
        bins=[-np.inf, 0.25, 0.45, np.inf],
        labels=["bajo", "medio", "alto"],
    ).astype("object").fillna("medio")

    return summary


def build_high_risk_matches(df: pd.DataFrame) -> pd.DataFrame:
    high = df[df["climate_risk"] == "alto"].copy()
    high = high.sort_values(
        ["climate_stress", "date", "Hora_UTC"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proyecta el Mundial 2026 usando ranking FIFA y normales climaticas de Open-Meteo."
    )
    parser.add_argument("--fixtures-path", default="partidos.csv", help="Ruta de partidos 2026.")
    parser.add_argument("--ranking-path", default="ranking_fifa_masculino.csv", help="Ruta ranking FIFA.")
    parser.add_argument("--out-csv-path", default="proyeccion_mundial_2026_clima.csv", help="CSV de salida.")
    parser.add_argument(
        "--group-summary-csv-path",
        default="resumen_clima_por_grupo_2026.csv",
        help="CSV de resumen climatico por grupo.",
    )
    parser.add_argument(
        "--high-risk-csv-path",
        default="partidos_alto_riesgo_climatico_2026.csv",
        help="CSV de partidos con alto riesgo climatico.",
    )
    parser.add_argument(
        "--climate-history-start",
        default="2020-01-01",
        help="Inicio del historico climatico (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--climate-history-end",
        default="2025-12-31",
        help="Fin del historico climatico (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        fixtures_path=args.fixtures_path,
        ranking_path=args.ranking_path,
        out_csv_path=args.out_csv_path,
        group_summary_csv_path=args.group_summary_csv_path,
        high_risk_csv_path=args.high_risk_csv_path,
        climate_history_start=args.climate_history_start,
        climate_history_end=args.climate_history_end,
    )

    fixtures = load_fixtures(cfg)
    ranking = load_ranking(cfg)

    normals = build_climate_normals(fixtures, cfg)
    if normals.empty:
        print("No se pudo construir normales climaticas.")
        return

    projection = build_2026_projection(fixtures, ranking, normals)

    missing_teams = projection[projection[["home_rank", "away_rank"]].isna().any(axis=1)]
    if not missing_teams.empty:
        print("\nAviso: algunos equipos no matchearon exacto con ranking (se uso mediana de puntos):")
        print(
            missing_teams[["home_team", "away_team"]]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )

    output_cols = [
        "Grupo",
        "date",
        "Hora_UTC",
        "home_team",
        "away_team",
        "city",
        "country",
        "temp_mean",
        "precip_sum",
        "wind_max",
        "climate_stress",
        "climate_risk",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "expected_total_goals",
        "climate_adjusted_goal_diff",
    ]

    projection.to_csv(cfg.out_csv_path, index=False, columns=output_cols)

    group_summary = build_group_summary(projection)
    group_summary.to_csv(cfg.group_summary_csv_path, index=False)

    high_risk = build_high_risk_matches(projection)
    high_risk_cols = [
        "Grupo",
        "date",
        "Hora_UTC",
        "home_team",
        "away_team",
        "city",
        "country",
        "temp_mean",
        "precip_sum",
        "wind_max",
        "climate_stress",
        "climate_risk",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "expected_total_goals",
    ]
    high_risk.to_csv(cfg.high_risk_csv_path, index=False, columns=high_risk_cols)

    print_projection_summary(projection)
    print(f"\nArchivo generado: {cfg.out_csv_path}")
    print(f"Archivo generado: {cfg.group_summary_csv_path}")
    print(f"Archivo generado: {cfg.high_risk_csv_path} (filas: {len(high_risk)})")


if __name__ == "__main__":
    main()
