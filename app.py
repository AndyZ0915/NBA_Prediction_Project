"""Streamlit dashboard for the NBA Player Development Predictor."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis import development_by_age, draft_analysis, player_summary, sophomore_jump
from src.data import build_transition_data, load_season_data, make_demo_data
from src.modeling import PlayerDevelopmentModels


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "all_seasons.csv"

st.set_page_config(
    page_title="NBA Player Development Predictor",
    layout="wide",
)


@st.cache_data

def load_project_data():
    """Load the real dataset, or use demo data when it is not present."""
    if DATA_PATH.exists():
        return load_season_data(DATA_PATH), False
    return make_demo_data(), True


@st.cache_resource

def train_dashboard_model(transitions: pd.DataFrame):
    """Train the small models once for the current dashboard session."""
    model = PlayerDevelopmentModels()
    model.fit(transitions)
    return model


def line_chart(data: pd.DataFrame, x: str, y: list[str], title: str, ylabel: str):
    """Draw a simple Matplotlib line chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for column in y:
        ax.plot(data[x], data[column], marker="o", label=column.upper())
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(y) > 1:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("NBA Player Development Predictor")
    st.write(
        "Explore historical player development and estimate next-season "
        "performance using previous-season statistics."
    )

    df, using_demo = load_project_data()

    if using_demo:
        st.info(
            "No all_seasons.csv was found in data/. The dashboard is using "
            "generated demo data. Add the real dataset for actual NBA results."
        )
    else:
        st.success("Using data/all_seasons.csv")

    try:
        transitions = build_transition_data(df)
    except ValueError as exc:
        st.error(f"The dataset does not contain enough consecutive seasons to build predictions: {exc}")
        st.stop()

    # The sidebar controls the player view while the rest of the dashboard
    # focuses on broader league-level analysis.
    players = sorted(transitions["player_name"].unique())
    selected_player = st.sidebar.selectbox("Player", players)

    history = (
        df[df["player_name"] == selected_player]
        .sort_values("season_year")
        .copy()
    )
    player_transitions = transitions[
        transitions["player_name"] == selected_player
    ].copy()

    st.header(selected_player)
    latest = history.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest PPG", f"{latest['pts']:.1f}")
    c2.metric("Latest RPG", f"{latest['reb']:.1f}")
    c3.metric("Latest APG", f"{latest['ast']:.1f}")
    c4.metric("Games", f"{latest['gp']:.0f}")

    st.subheader("Player history")
    line_chart(
        history,
        "season_year",
        ["pts", "reb", "ast"],
        "Player production by season",
        "Per-game average",
    )

    if not player_transitions.empty:
        st.subheader("Year-to-year development")
        display = player_transitions[
            ["target_season", "prev_ppg", "next_ppg", "ppg_change", "ppg_change_pct"]
        ].rename(
            columns={
                "target_season": "Season",
                "prev_ppg": "Previous PPG",
                "next_ppg": "Next PPG",
                "ppg_change": "PPG Change",
                "ppg_change_pct": "PPG Change %",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.header("Prediction")
    try:
        model = train_dashboard_model(transitions)
        result = model.predict_player(player_transitions.iloc[-1])

        p1, p2, p3 = st.columns(3)
        p1.metric("Projected next-season PPG", f"{result['predicted_ppg']:.1f}")
        p2.metric("15%+ improvement probability", f"{result['improvement_probability']:.0%}")
        p3.metric("Previous PPG", f"{player_transitions.iloc[-1]['prev_ppg']:.1f}")

        st.caption(
            "This is a historical prediction exercise. The estimate is not a "
            "guarantee and does not account for every factor affecting a player."
        )
    except Exception as exc:
        st.warning(f"A model prediction could not be generated: {exc}")
        model = None

    st.header("League analysis")

    tab1, tab2, tab3 = st.tabs(["Development", "Player comparisons", "Model details"])

    with tab1:
        age = development_by_age(df)
        st.subheader("Scoring by age")
        line_chart(age, "age", ["ppg"], "Average PPG by age", "Points per game")

        st.subheader("Rookie to sophomore change")
        jump = sophomore_jump(df)
        if not jump.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(jump["change_pct"].dropna(), bins=20)
            ax.axvline(0, linestyle="--")
            ax.set_title("Rookie to sophomore PPG change")
            ax.set_xlabel("PPG change (%)")
            ax.set_ylabel("Players")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Games played")
        availability = (
            df.groupby("player_name")
            .agg(avg_games=("gp", "mean"), seasons=("season_year", "nunique"))
        )
        availability = availability[availability["seasons"] >= 3]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(availability["avg_games"], bins=20)
        ax.set_title("Average games played for multi-season players")
        ax.set_xlabel("Average games per season")
        ax.set_ylabel("Players")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Draft tier comparison")
        draft = draft_analysis(df)
        st.bar_chart(draft.set_index("draft_tier")["avg_ppg"])
        st.dataframe(draft, use_container_width=True, hide_index=True)

        st.subheader("Player summary")
        summary = player_summary(df)
        st.dataframe(summary.head(25), use_container_width=True, hide_index=True)

    with tab3:
        if model is not None:
            st.subheader("Test metrics")
            metrics = {
                **model.regression_metrics,
                **model.classification_metrics,
            }
            st.dataframe(
                pd.DataFrame(
                    [{"Metric": key, "Value": value} for key, value in metrics.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Feature importance")
            st.dataframe(
                model.feature_importance(),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Recent test predictions")
            st.dataframe(
                model.test_predictions.head(20),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
