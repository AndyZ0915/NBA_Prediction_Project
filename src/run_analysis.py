"""Run the complete analysis from the command line."""

from pathlib import Path
import json
import pandas as pd

from .analysis import development_by_age, draft_analysis, player_summary, sophomore_jump
from .data import build_transition_data, load_season_data
from .database import NBADatabase, run_sql_examples
from .modeling import PlayerDevelopmentModels
from .visuals import make_charts


def main(data_path: str = "data/all_seasons.csv") -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"
    model_dir = output_dir / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    df = load_season_data(root / data_path)
    transitions = build_transition_data(df)

    # Keep the database part separate from the ML data so both workflows can
    # be inspected without making the project depend on one another.
    db = NBADatabase(output_dir / "nba_analytics.db")
    db.connect()
    db.create_schema()
    db.load_data(df)
    sql_results = run_sql_examples(db)
    for name, table in sql_results.items():
        table.to_csv(output_dir / f"sql_{name}.csv", index=False)
    db.close()

    transitions.to_csv(output_dir / "player_transitions.csv", index=False)
    player_summary(df).to_csv(output_dir / "player_summary.csv", index=False)
    development_by_age(df).to_csv(output_dir / "development_by_age.csv", index=False)
    draft_analysis(df).to_csv(output_dir / "draft_analysis.csv", index=False)
    sophomore_jump(df).to_csv(output_dir / "sophomore_jump.csv", index=False)

    model = PlayerDevelopmentModels()
    split = model.fit(transitions)
    model.test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    model.feature_importance().to_csv(output_dir / "feature_importance.csv", index=False)
    model.save(model_dir)

    charts = make_charts(df, transitions, output_dir / "charts")
    metrics = {
        "rows": int(len(df)),
        "player_season_transitions": int(len(transitions)),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "regression": model.regression_metrics,
        "classification": model.classification_metrics,
        "charts": charts,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Analysis complete.")
    print(f"Rows: {len(df):,}")
    print(f"Transitions: {len(transitions):,}")
    print(f"Test PPG MAE: {model.regression_metrics['test_mae']:.2f}")
    print(f"Test PPG R²: {model.regression_metrics['test_r2']:.3f}")
    print(f"Improvement accuracy: {model.classification_metrics['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
