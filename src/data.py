"""Data loading, cleaning, and feature preparation for the NBA project."""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "player_name", "season", "age", "gp", "pts", "reb", "ast",
    "player_height", "player_weight", "draft_year", "draft_round",
    "draft_number", "ts_pct", "usg_pct", "net_rating", "oreb_pct",
    "dreb_pct", "ast_pct", "team_abbreviation"
]


def load_season_data(path: str | Path) -> pd.DataFrame:
    """Load the season-level CSV and apply basic cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put all_seasons.csv in the data folder."
        )

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    numeric_columns = [
        "age", "gp", "pts", "reb", "ast", "player_height", "player_weight",
        "draft_year", "draft_round", "draft_number", "ts_pct", "usg_pct",
        "net_rating", "oreb_pct", "dreb_pct", "ast_pct"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["college"] = df.get("college", pd.Series(index=df.index)).fillna("Not specified")
    df["country"] = df.get("country", pd.Series(index=df.index)).fillna("Not specified")
    df["season_year"] = df["season"].astype(str).str[:4].astype(int)

    df = df.dropna(subset=["player_name", "season", "pts", "gp", "age"])
    df = df.sort_values(["player_name", "season_year"]).reset_index(drop=True)
    return df


def build_transition_data(df: pd.DataFrame, minimum_games: int = 20) -> pd.DataFrame:
    """Create one row per player-to-next-season transition."""
    work = df[df["gp"] >= minimum_games].copy()
    work = work.sort_values(["player_name", "season_year"])

    rows = []
    for player, group in work.groupby("player_name"):
        group = group.sort_values("season_year").reset_index(drop=True)
        for i in range(1, len(group)):
            previous = group.iloc[i - 1]
            current = group.iloc[i]

            # A missing season usually means the player was not active, so this
            # is not a clean one-year development comparison.
            if current["season_year"] - previous["season_year"] != 1:
                continue

            earlier = group.iloc[:i]
            rows.append({
                "player_name": player,
                "target_season": current["season"],
                "target_year": int(current["season_year"]),
                "age": previous["age"],
                "seasons_in_league": i,
                "prev_ppg": previous["pts"],
                "prev_rpg": previous["reb"],
                "prev_apg": previous["ast"],
                "prev_gp": previous["gp"],
                "prev_ts_pct": previous["ts_pct"],
                "prev_usg_pct": previous["usg_pct"],
                "prev_net_rating": previous["net_rating"],
                "prev_oreb_pct": previous["oreb_pct"],
                "prev_dreb_pct": previous["dreb_pct"],
                "prev_ast_pct": previous["ast_pct"],
                "career_ppg": earlier["pts"].mean(),
                "career_gp": earlier["gp"].mean(),
                "height": previous["player_height"],
                "weight": previous["player_weight"],
                "draft_number": previous["draft_number"],
                "draft_round": previous["draft_round"],
                "next_ppg": current["pts"],
                "next_rpg": current["reb"],
                "next_apg": current["ast"],
                "next_gp": current["gp"],
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No valid player-season transitions were created.")

    result["ppg_change"] = result["next_ppg"] - result["prev_ppg"]
    result["ppg_change_pct"] = 100 * result["ppg_change"] / result["prev_ppg"].clip(lower=0.1)
    result["improved_15pct"] = (result["ppg_change_pct"] >= 15).astype(int)
    result["draft_tier"] = pd.cut(
        result["draft_number"].fillna(61),
        bins=[0, 5, 15, 30, 60, float("inf")],
        labels=["Top 5", "Lottery", "Late 1st", "2nd Round", "Undrafted/Unknown"],
        include_lowest=True,
    ).astype(str)
    return result


def feature_columns() -> list[str]:
    return [
        "age", "seasons_in_league", "prev_ppg", "prev_rpg", "prev_apg",
        "prev_gp", "prev_ts_pct", "prev_usg_pct", "prev_net_rating",
        "prev_oreb_pct", "prev_dreb_pct", "prev_ast_pct", "career_ppg",
        "career_gp", "height", "weight", "draft_number", "draft_round"
    ]


# Backwards-compatible names used by earlier dashboard versions.
def load_csv(path: str | Path) -> pd.DataFrame:
    """Compatibility wrapper for older project entry points."""
    return load_season_data(path)


def make_demo_data(rows: int = 0) -> pd.DataFrame:
    """Create a small reproducible dataset for testing the project locally.

    This is only a fallback for development. Real analysis should use
    the supplied NBA season dataset in data/all_seasons.csv.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    seasons = list(range(2015, 2024))
    players = [f"Demo Player {i:03d}" for i in range(1, 101)]
    records = []
    for player in players:
        age_start = int(rng.integers(19, 26))
        base = float(rng.uniform(5, 24))
        for j, year in enumerate(seasons):
            if rng.random() < 0.08 and j > 0:
                continue
            age = age_start + j
            pts = max(2.0, base + j * rng.normal(0.7, 0.35) + rng.normal(0, 2.0))
            reb = max(1.0, rng.normal(5.5, 2.0))
            ast = max(0.5, rng.normal(3.5, 1.8))
            records.append({
                "player_name": player, "season": f"{year}-{str(year+1)[-2:]}",
                "age": age, "gp": int(rng.integers(25, 83)),
                "pts": pts, "reb": reb, "ast": ast,
                "player_height": float(rng.normal(198, 10)),
                "player_weight": float(rng.normal(95, 12)),
                "draft_year": age_start + 19, "draft_round": int(rng.integers(1, 3)),
                "draft_number": int(rng.integers(1, 61)),
                "ts_pct": float(rng.uniform(48, 66)),
                "usg_pct": float(rng.uniform(12, 30)),
                "net_rating": float(rng.normal(0, 8)),
                "oreb_pct": float(rng.uniform(1, 10)),
                "dreb_pct": float(rng.uniform(8, 25)),
                "ast_pct": float(rng.uniform(5, 30)),
                "team_abbreviation": "DEMO",
                "college": "Demo College", "country": "USA"
            })
    demo = pd.DataFrame(records)
    demo["season_year"] = demo["season"].astype(str).str[:4].astype(int)
    return demo
