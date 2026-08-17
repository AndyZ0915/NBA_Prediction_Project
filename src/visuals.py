"""Reusable Matplotlib charts used by the analysis script."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_chart(fig, output_dir: str | Path, name: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_charts(df: pd.DataFrame, transitions: pd.DataFrame, output_dir: str | Path) -> list[str]:
    paths = []

    age = df[df.gp >= 20].groupby("age")["pts"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(age.index, age.values, marker="o")
    ax.set(title="Average Scoring by Age", xlabel="Age", ylabel="Points per game")
    ax.grid(alpha=0.25)
    paths.append(save_chart(fig, output_dir, "scoring_by_age.png"))

    jump = transitions[transitions.seasons_in_league == 1]["ppg_change_pct"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jump.dropna(), bins=25)
    ax.axvline(0, linestyle="--")
    ax.set(title="Rookie to Second-Year Change", xlabel="PPG change (%)", ylabel="Players")
    paths.append(save_chart(fig, output_dir, "sophomore_jump.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(transitions.prev_ppg, transitions.next_ppg, alpha=0.35)
    low = min(transitions.prev_ppg.min(), transitions.next_ppg.min())
    high = max(transitions.prev_ppg.max(), transitions.next_ppg.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(title="Year-to-Year Scoring Consistency", xlabel="Previous season PPG", ylabel="Next season PPG")
    ax.grid(alpha=0.25)
    paths.append(save_chart(fig, output_dir, "scoring_consistency.png"))

    draft = df.copy()
    draft["draft_tier"] = pd.cut(
        draft.draft_number.fillna(61), [0, 5, 15, 30, 60, float("inf")],
        labels=["Top 5", "Lottery", "Late 1st", "2nd Round", "Undrafted/Unknown"], include_lowest=True
    )
    draft_avg = draft.groupby("draft_tier", observed=False).pts.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    draft_avg.plot(kind="bar", ax=ax)
    ax.set(title="Average PPG by Draft Tier", xlabel="Draft tier", ylabel="Points per game")
    ax.tick_params(axis="x", rotation=20)
    paths.append(save_chart(fig, output_dir, "draft_tier_scoring.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(transitions.prev_ppg, transitions.ppg_change, alpha=0.35)
    ax.axhline(0, linestyle="--")
    ax.set(title="Who Improves the Most?", xlabel="Previous season PPG", ylabel="Next-season PPG change")
    ax.grid(alpha=0.25)
    paths.append(save_chart(fig, output_dir, "improvement_by_previous_ppg.png"))

    durability = df.groupby("player_name").agg(avg_gp=("gp", "mean"), seasons=("season_year", "nunique"))
    durability = durability[durability.seasons >= 3]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durability.avg_gp, bins=25)
    ax.set(title="Typical Games Played for Multi-Season Players", xlabel="Average games per season", ylabel="Players")
    paths.append(save_chart(fig, output_dir, "player_availability.png"))

    return paths
