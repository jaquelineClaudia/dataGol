"""
simulacion_bracket.py
====================
Simula el bracket completo del Mundial 2026 usando:
- Grupos reales y cruces reales del fixture 2026
- Probabilidades del modelo (probabilidades_ganador_2026.csv)

Genera:
- bracket_simulado.csv
- probabilidades_por_fase.csv

Uso:
  python simulacion_bracket.py
  python simulacion_bracket.py --only-bracket
  python simulacion_bracket.py --n-sims 10000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
GROUP_ORDER = [chr(c) for c in range(ord("A"), ord("L") + 1)]
THIRD_TOKEN_PATTERN = re.compile(r"^3([A-L](?:/[A-L])*)$")

TEAM_ALIASES = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Curacao": "Curaçao",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Turkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Congo DR": "DR Congo",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        signature = f.read(2)
    if signature == b"PK":
        return pd.read_excel(path)

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
    return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


def norm_team(name: str) -> str:
    raw = str(name).strip()
    return TEAM_ALIASES.get(raw, raw)


def build_strength_map(group_fixtures: pd.DataFrame) -> Dict[str, float]:
    prob_df = read_csv_auto(BASE_DIR / "probabilidades_ganador_2026.csv")
    ranking_df = read_csv_auto(BASE_DIR / "ranking_fifa_masculino.csv")
    market_df = read_csv_auto(BASE_DIR / "valor_mercado_2026.csv")

    if "proba_final" not in prob_df.columns and "proba_final_%" in prob_df.columns:
        prob_df["proba_final"] = (
            prob_df["proba_final_%"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
        )

    prob_map: Dict[str, float] = {}
    for _, row in prob_df.iterrows():
        team = norm_team(row["equipo"])
        prob_map[team] = float(row["proba_final"])

    ranking_map: Dict[str, float] = {}
    for _, row in ranking_df.iterrows():
        team = norm_team(row["pais"])
        ranking_map[team] = float(row["puntos"])

    market_map: Dict[str, float] = {}
    for _, row in market_df.iterrows():
        team = norm_team(row["equipo"])
        market_map[team] = float(row["valor_mercado_millones_eur"])

    teams = sorted(set(group_fixtures["Equipo_1"].map(norm_team)).union(set(group_fixtures["Equipo_2"].map(norm_team))))

    prob_vals = np.array([prob_map.get(t, np.nan) for t in teams], dtype=float)
    rank_vals = np.array([ranking_map.get(t, np.nan) for t in teams], dtype=float)
    market_vals = np.array([market_map.get(t, np.nan) for t in teams], dtype=float)

    prob_med = np.nanmedian(prob_vals) if not np.isnan(prob_vals).all() else 0.03
    rank_med = np.nanmedian(rank_vals) if not np.isnan(rank_vals).all() else 1400.0
    market_med = np.nanmedian(market_vals) if not np.isnan(market_vals).all() else 120.0

    prob_vals = np.where(np.isnan(prob_vals), prob_med, prob_vals)
    rank_vals = np.where(np.isnan(rank_vals), rank_med, rank_vals)
    market_vals = np.where(np.isnan(market_vals), market_med, market_vals)

    def minmax(arr: np.ndarray) -> np.ndarray:
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-9)

    prob_norm = minmax(prob_vals)
    rank_norm = minmax(rank_vals)
    market_norm = minmax(market_vals)

    strength: Dict[str, float] = {}
    for idx, team in enumerate(teams):
        # Modelo ML domina; ranking y mercado estabilizan faltantes.
        strength[team] = float(0.70 * prob_norm[idx] + 0.20 * rank_norm[idx] + 0.10 * market_norm[idx])
    return strength


def match_outcome_probs(team_a: str, team_b: str, strength: Dict[str, float], knockout: bool) -> Tuple[float, float, float]:
    sa = strength.get(team_a, 0.5)
    sb = strength.get(team_b, 0.5)

    p_a_raw = sa / (sa + sb + 1e-9)
    gap = abs(p_a_raw - 0.5) * 2.0
    p_draw = 0.0 if knockout else float(np.clip(0.24 - 0.12 * gap, 0.08, 0.26))

    rem = 1.0 - p_draw
    p_a = rem * p_a_raw
    p_b = rem * (1.0 - p_a_raw)
    return p_a, p_draw, p_b


def expected_goals(team_a: str, team_b: str, strength: Dict[str, float]) -> Tuple[float, float]:
    sa = strength.get(team_a, 0.5)
    sb = strength.get(team_b, 0.5)
    diff = sa - sb
    base = 2.55
    lam_a = np.clip(base * (0.5 + 0.55 * diff), 0.35, 3.20)
    lam_b = np.clip(base * (0.5 - 0.55 * diff), 0.35, 3.20)
    return float(lam_a), float(lam_b)


def simulate_match(
    team_a: str,
    team_b: str,
    strength: Dict[str, float],
    rng: np.random.Generator,
    knockout: bool,
) -> Tuple[int, int, str, str]:
    p_a, p_d, p_b = match_outcome_probs(team_a, team_b, strength, knockout=knockout)
    outcome = rng.choice(["A", "D", "B"], p=[p_a, p_d, p_b])

    lam_a, lam_b = expected_goals(team_a, team_b, strength)
    ga = int(rng.poisson(lam_a))
    gb = int(rng.poisson(lam_b))

    if outcome == "A" and ga <= gb:
        ga = gb + 1
    elif outcome == "B" and gb <= ga:
        gb = ga + 1
    elif outcome == "D":
        gb = ga

    if knockout and ga == gb:
        sa = strength.get(team_a, 0.5)
        sb = strength.get(team_b, 0.5)
        p_pen_a = float(np.clip(0.50 + 0.10 * (sa - sb), 0.35, 0.65))
        if rng.random() < p_pen_a:
            ga += 1
            return ga, gb, team_a, "penales"
        gb += 1
        return ga, gb, team_b, "penales"

    winner = team_a if ga > gb else team_b if gb > ga else ""
    method = "90min"
    return ga, gb, winner, method


def simulate_group_phase(
    group_fixtures: pd.DataFrame,
    strength: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[str]], Dict[str, str], List[dict], List[dict]]:
    standings_by_group: Dict[str, List[str]] = {}
    third_by_group: Dict[str, str] = {}
    third_rank_rows: List[dict] = []
    group_match_rows: List[dict] = []

    for group in GROUP_ORDER:
        gf = group_fixtures[group_fixtures["Grupo"] == group].copy()
        if gf.empty:
            continue

        teams = sorted(set(gf["Equipo_1"].map(norm_team)).union(set(gf["Equipo_2"].map(norm_team))))
        stats = {
            t: {
                "pts": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "wins": 0,
            }
            for t in teams
        }

        for _, row in gf.iterrows():
            t1 = norm_team(row["Equipo_1"])
            t2 = norm_team(row["Equipo_2"])
            g1, g2, winner, method = simulate_match(t1, t2, strength, rng, knockout=False)

            stats[t1]["gf"] += g1
            stats[t1]["ga"] += g2
            stats[t1]["gd"] += g1 - g2
            stats[t2]["gf"] += g2
            stats[t2]["ga"] += g1
            stats[t2]["gd"] += g2 - g1

            if g1 > g2:
                stats[t1]["pts"] += 3
                stats[t1]["wins"] += 1
            elif g2 > g1:
                stats[t2]["pts"] += 3
                stats[t2]["wins"] += 1
            else:
                stats[t1]["pts"] += 1
                stats[t2]["pts"] += 1

            group_match_rows.append(
                {
                    "fase": "GRUPOS",
                    "match_id": "",
                    "grupo": group,
                    "fecha": row.get("Fecha", ""),
                    "equipo_1": t1,
                    "equipo_2": t2,
                    "goles_1": g1,
                    "goles_2": g2,
                    "ganador": winner,
                    "metodo": method,
                    "slot_1": t1,
                    "slot_2": t2,
                    "sede": row.get("Sede", ""),
                }
            )

        ranked = sorted(
            teams,
            key=lambda t: (
                stats[t]["pts"],
                stats[t]["gd"],
                stats[t]["gf"],
                stats[t]["wins"],
                strength.get(t, 0.0),
                rng.random() * 1e-6,
            ),
            reverse=True,
        )
        standings_by_group[group] = ranked
        third_by_group[group] = ranked[2]
        t3 = ranked[2]

        third_rank_rows.append(
            {
                "grupo": group,
                "equipo": t3,
                "pts": stats[t3]["pts"],
                "gd": stats[t3]["gd"],
                "gf": stats[t3]["gf"],
                "strength": strength.get(t3, 0.0),
            }
        )

    third_rank_rows = sorted(
        third_rank_rows,
        key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]),
        reverse=True,
    )

    return standings_by_group, third_by_group, group_match_rows, third_rank_rows


def assign_third_place_tokens(
    third_slots: List[str],
    third_by_group: Dict[str, str],
    third_rank_rows: List[dict],
) -> Dict[str, str]:
    rank_pos = {row["equipo"]: i for i, row in enumerate(third_rank_rows)}

    slot_groups: Dict[str, List[str]] = {}
    for slot in third_slots:
        token = slot.strip()
        m = THIRD_TOKEN_PATTERN.match(token)
        if not m:
            continue
        slot_groups[token] = m.group(1).split("/")

    # Backtracking para respetar restricciones de cada slot de terceros.
    tokens_sorted = sorted(slot_groups.keys(), key=lambda x: len(slot_groups[x]))

    def solve(idx: int, used_teams: set, out: Dict[str, str]) -> Dict[str, str] | None:
        if idx >= len(tokens_sorted):
            return dict(out)

        token = tokens_sorted[idx]
        candidate_teams = []
        for g in slot_groups[token]:
            team = third_by_group.get(g)
            if team and team not in used_teams:
                candidate_teams.append(team)

        candidate_teams = sorted(set(candidate_teams), key=lambda t: rank_pos.get(t, 999))

        for team in candidate_teams:
            out[token] = team
            used_teams.add(team)
            solved = solve(idx + 1, used_teams, out)
            if solved is not None:
                return solved
            used_teams.remove(team)
            out.pop(token, None)

        return None

    solved_map = solve(0, set(), {})
    if solved_map is None:
        solved_map = {}

    return solved_map


def simulate_knockout_phase(
    phase_fixtures: pd.DataFrame,
    standings_by_group: Dict[str, List[str]],
    third_by_group: Dict[str, str],
    third_rank_rows: List[dict],
    strength: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[List[dict], Dict[str, dict], str]:
    phase_df = phase_fixtures[phase_fixtures["Grupo"] != "3P"].copy().reset_index(drop=True)
    phase_df["match_id"] = np.arange(73, 73 + len(phase_df))

    third_tokens = []
    for _, row in phase_df.iterrows():
        for token in (str(row["Equipo_1"]).strip(), str(row["Equipo_2"]).strip()):
            if THIRD_TOKEN_PATTERN.match(token):
                third_tokens.append(token)

    third_token_map = assign_third_place_tokens(sorted(set(third_tokens)), third_by_group, third_rank_rows)

    winners_by_match: Dict[str, str] = {}
    losers_by_match: Dict[str, str] = {}
    knockout_rows: List[dict] = []

    reached = {}
    for teams in standings_by_group.values():
        for t in teams:
            reached[t] = {
                "group_winner": False,
                "knockout": False,
                "octavos": False,
                "cuartos": False,
                "semis": False,
                "final": False,
                "campeon": False,
            }

    for group, ranked in standings_by_group.items():
        reached[ranked[0]]["group_winner"] = True

    third_qualifiers = {r["equipo"] for r in third_rank_rows[:8]}
    for group, ranked in standings_by_group.items():
        for t in ranked[:2]:
            reached[t]["knockout"] = True
        if ranked[2] in third_qualifiers:
            reached[ranked[2]]["knockout"] = True

    def resolve_slot(slot: str) -> str:
        token = slot.strip()
        if token in third_token_map:
            return third_token_map[token]

        m_pos = re.match(r"^([12])([A-L])$", token)
        if m_pos:
            pos = int(m_pos.group(1)) - 1
            grp = m_pos.group(2)
            return standings_by_group[grp][pos]

        m_third = re.match(r"^3([A-L])$", token)
        if m_third:
            return third_by_group[m_third.group(1)]

        m_w = re.match(r"^W(\d+)$", token)
        if m_w:
            return winners_by_match[token]

        m_l = re.match(r"^L(\d+)$", token)
        if m_l:
            return losers_by_match[token]

        return norm_team(token)

    for _, row in phase_df.iterrows():
        stage = str(row["Grupo"]).strip()
        match_id = int(row["match_id"])
        slot_1 = str(row["Equipo_1"]).strip()
        slot_2 = str(row["Equipo_2"]).strip()

        team_1 = resolve_slot(slot_1)
        team_2 = resolve_slot(slot_2)

        g1, g2, winner, method = simulate_match(team_1, team_2, strength, rng, knockout=True)
        loser = team_2 if winner == team_1 else team_1

        winners_by_match[f"W{match_id}"] = winner
        losers_by_match[f"L{match_id}"] = loser

        if stage == "R32":
            reached[winner]["octavos"] = True
        elif stage == "R16":
            reached[winner]["cuartos"] = True
        elif stage == "QF":
            reached[winner]["semis"] = True
        elif stage == "SF":
            reached[winner]["final"] = True
        elif stage == "F":
            reached[winner]["campeon"] = True

        knockout_rows.append(
            {
                "fase": stage,
                "match_id": match_id,
                "grupo": "",
                "fecha": row.get("Fecha", ""),
                "equipo_1": team_1,
                "equipo_2": team_2,
                "goles_1": g1,
                "goles_2": g2,
                "ganador": winner,
                "metodo": method,
                "slot_1": slot_1,
                "slot_2": slot_2,
                "sede": row.get("Sede", ""),
            }
        )

    champion_row = next((r for r in knockout_rows if r["fase"] == "F"), None)
    champion = champion_row["ganador"] if champion_row else ""

    return knockout_rows, reached, champion


def simulate_single_world_cup(
    group_fixtures: pd.DataFrame,
    phase_fixtures: pd.DataFrame,
    strength: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, dict], str]:
    standings_by_group, third_by_group, group_rows, third_rank_rows = simulate_group_phase(group_fixtures, strength, rng)
    knockout_rows, reached, champion = simulate_knockout_phase(
        phase_fixtures,
        standings_by_group,
        third_by_group,
        third_rank_rows,
        strength,
        rng,
    )

    bracket_rows = group_rows + knockout_rows
    bracket_df = pd.DataFrame(bracket_rows)

    return bracket_df, reached, champion


def run_monte_carlo(
    group_fixtures: pd.DataFrame,
    phase_fixtures: pd.DataFrame,
    strength: Dict[str, float],
    n_sims: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    teams = sorted(set(group_fixtures["Equipo_1"].map(norm_team)).union(set(group_fixtures["Equipo_2"].map(norm_team))))

    counters = {
        t: {
            "group_winner": 0,
            "knockout": 0,
            "octavos": 0,
            "cuartos": 0,
            "semis": 0,
            "final": 0,
            "campeon": 0,
        }
        for t in teams
    }

    for idx in range(n_sims):
        _, reached, _ = simulate_single_world_cup(group_fixtures, phase_fixtures, strength, rng)
        for team in teams:
            stage = reached[team]
            counters[team]["group_winner"] += int(stage["group_winner"])
            counters[team]["knockout"] += int(stage["knockout"])
            counters[team]["octavos"] += int(stage["octavos"])
            counters[team]["cuartos"] += int(stage["cuartos"])
            counters[team]["semis"] += int(stage["semis"])
            counters[team]["final"] += int(stage["final"])
            counters[team]["campeon"] += int(stage["campeon"])

        if (idx + 1) % 2000 == 0:
            print(f"  Simulaciones: {idx + 1:,}/{n_sims:,}")

    rows = []
    for team in teams:
        c = counters[team]
        rows.append(
            {
                "Equipo": team,
                "P_Gana_Grupo_%": round(100.0 * c["group_winner"] / n_sims, 2),
                "P_Llega_Octavos_%": round(100.0 * c["octavos"] / n_sims, 2),
                "P_Llega_Cuartos_%": round(100.0 * c["cuartos"] / n_sims, 2),
                "P_Llega_Semis_%": round(100.0 * c["semis"] / n_sims, 2),
                "P_Llega_Final_%": round(100.0 * c["final"] / n_sims, 2),
                "P_Gana_Mundial_%": round(100.0 * c["campeon"] / n_sims, 2),
                "P_Clasifica_Eliminatoria_%": round(100.0 * c["knockout"] / n_sims, 2),
            }
        )

    out = pd.DataFrame(rows).sort_values("P_Gana_Mundial_%", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulacion de bracket completo Mundial 2026")
    parser.add_argument("--n-sims", type=int, default=10_000, help="Numero de simulaciones Monte Carlo")
    parser.add_argument("--only-bracket", action="store_true", help="Solo genera un bracket aleatorio")
    args = parser.parse_args()

    print("=" * 68)
    print("  BRACKET MUNDIAL 2026 - SIMULACION PARTIDO A PARTIDO")
    print("=" * 68)

    all_fixtures = read_csv_auto(BASE_DIR / "partidos_completo_2026.csv")
    group_fixtures = all_fixtures[all_fixtures["Grupo"].isin(GROUP_ORDER)].copy()

    # Excluye filas contaminadas con placeholders de eliminatoria dentro de grupos.
    slot_mask = group_fixtures["Equipo_1"].astype(str).str.match(r"^[WwLl123]") | \
        group_fixtures["Equipo_2"].astype(str).str.match(r"^[WwLl123]")
    group_fixtures = group_fixtures[~slot_mask].copy()
    phase_fixtures = read_csv_auto(BASE_DIR / "partidos_fase_final_2026.csv")

    for col in ("Equipo_1", "Equipo_2"):
        group_fixtures[col] = group_fixtures[col].map(norm_team)
        phase_fixtures[col] = phase_fixtures[col].map(lambda x: norm_team(x) if not str(x).startswith(("1", "2", "3", "W", "L")) else str(x))

    strength = build_strength_map(group_fixtures)
    rng = np.random.default_rng()

    bracket_df, _, champion = simulate_single_world_cup(group_fixtures, phase_fixtures, strength, rng)
    bracket_df.to_csv(BASE_DIR / "bracket_simulado.csv", index=False)

    print(f"  Bracket simulado guardado en bracket_simulado.csv ({len(bracket_df)} partidos).")
    print(f"  Campeon simulado: {champion}")

    if args.only_bracket:
        print("  Modo only-bracket: se conserva probabilidades_por_fase.csv existente.")
        return

    print(f"\n  Ejecutando Monte Carlo de bracket: {args.n_sims:,} simulaciones...")
    probs = run_monte_carlo(group_fixtures, phase_fixtures, strength, args.n_sims, rng)
    probs.to_csv(BASE_DIR / "probabilidades_por_fase.csv", index=True)

    print("\n  Top 10 probabilidad de campeon:")
    print(probs[["Equipo", "P_Gana_Mundial_%"]].head(10).to_string(index=True))
    print("\n  Archivo guardado: probabilidades_por_fase.csv")


if __name__ == "__main__":
    main()
