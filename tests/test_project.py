import pandas as pd

from src.data import build_transition_data
from src.modeling import PlayerDevelopmentModels


def sample_data():
    rows = []
    for player_id in range(12):
        for year in range(2015, 2022):
            rows.append({
                "player_name": f"Player {player_id}",
                "season": f"{year}-{str(year + 1)[-2:]}",
                "season_year": year,
                "age": 20 + (year - 2015) * 0.8,
                "gp": 60,
                "pts": 8 + player_id + (year - 2015) * (1.5 if player_id % 3 == 0 else 0.3),
                "reb": 3 + player_id * 0.1,
                "ast": 2 + player_id * 0.1,
                "player_height": 195,
                "player_weight": 90,
                "draft_year": 2015,
                "draft_round": 1,
                "draft_number": 10 + player_id,
                "ts_pct": 0.55,
                "usg_pct": 20,
                "net_rating": 2,
                "oreb_pct": 0.05,
                "dreb_pct": 0.15,
                "ast_pct": 15,
            })
    return pd.DataFrame(rows)


def test_transition_and_model_split():
    transitions = build_transition_data(sample_data())
    assert len(transitions) > 0
    model = PlayerDevelopmentModels()
    split = model.temporal_split(transitions)
    assert split.train.target_year.max() < split.validation.target_year.min()
    assert split.validation.target_year.max() < split.test.target_year.min()
