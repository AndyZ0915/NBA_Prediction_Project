"""Exploratory analysis and player-level summaries."""

import pandas as pd


def player_history(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    return df[df["player_name"] == player_name].sort_values("season_year").copy()


def player_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("player_name")
        .agg(
            seasons=("season_year", "nunique"),
            avg_ppg=("pts", "mean"),
            avg_rpg=("reb", "mean"),
            avg_apg=("ast", "mean"),
            avg_gp=("gp", "mean"),
            best_ppg=("pts", "max"),
        )
        .reset_index()
        .sort_values("avg_ppg", ascending=False)
    )


def development_by_age(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["gp"] >= 20]
        .groupby("age", as_index=False)
        .agg(ppg=("pts", "mean"), rpg=("reb", "mean"), apg=("ast", "mean"), players=("player_name", "nunique"))
        .sort_values("age")
    )


def draft_analysis(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["draft_tier"] = pd.cut(
        work["draft_number"].fillna(61),
        bins=[0, 5, 15, 30, 60, float("inf")],
        labels=["Top 5", "Lottery", "Late 1st", "2nd Round", "Undrafted/Unknown"],
        include_lowest=True,
    )
    return (
        work.groupby("draft_tier", observed=False)
        .agg(avg_ppg=("pts", "mean"), avg_gp=("gp", "mean"), players=("player_name", "nunique"))
        .reset_index()
    )


def sophomore_jump(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["player_name", "season_year"]).copy()
    work["career_year"] = work.groupby("player_name").cumcount() + 1
    rookies = work[work["career_year"] == 1][["player_name", "pts"]].rename(columns={"pts": "rookie_ppg"})
    sophomores = work[work["career_year"] == 2][["player_name", "pts"]].rename(columns={"pts": "sophomore_ppg"})
    result = rookies.merge(sophomores, on="player_name")
    result["change_pct"] = 100 * (result["sophomore_ppg"] - result["rookie_ppg"]) / result["rookie_ppg"].clip(lower=0.1)
    return result
