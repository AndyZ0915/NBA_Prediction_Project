# NBA Player Development Predictor

A small NBA analytics project that studies how player performance changes from one season to the next and tests whether historical statistics can help predict future performance.

I built this project to practice a complete data workflow: cleaning season-level data, storing it in SQLite, answering questions with SQL, exploring trends with visualizations, creating time-aware features, training models, and presenting the results in a simple Streamlit dashboard.

## What it looks at

- How scoring changes as players age
- Rookie to sophomore development
- Year-to-year scoring consistency
- Draft tier and player production
- Player availability and games played
- Whether previous-season statistics help predict next-season PPG
- Which players have a higher probability of improving by at least 15%
- How the models compare with a simple previous-season baseline

## Models

### Next-season PPG

A Random Forest regressor estimates a player's next-season points per game using previous-season statistics and earlier career information.

The project reports:

- MAE
- R²
- Previous-season PPG baseline

### Improvement classification

A Random Forest classifier estimates whether a player will improve their PPG by at least 15% in the following season.

The project reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

The models are meant to be an analysis exercise, not a professional scouting or betting system.

## Train, validation, and test setup

The project uses a temporal split rather than randomly splitting player-season rows.

Older target seasons are used for training, the next two seasons are used for validation, and the final two seasons are used for testing. This keeps future seasons out of the training data and better matches the question the project is asking.

Missing numeric values are filled using medians calculated from the training set only.

## SQL database

The SQLite database contains:

```text
players
    |
    +---- player_stats ---- teams
```

The SQL examples use joins, grouping, filtering, and a `LAG()` window function to compare player seasons.

## Visual analysis

The analysis produces charts for:

1. Average scoring by age
2. Rookie to sophomore scoring changes
3. Year-to-year scoring consistency
4. Average scoring by draft tier
5. PPG improvement versus previous-season scoring
6. Typical games played for multi-season players

The Streamlit dashboard also lets you select an individual player and view their history, development, prediction, and related league analysis.

## Project structure

```text
NBA_Player_Development_Predictor/
├── app.py
├── app/
│   └── dashboard.py
├── data/
│   └── all_seasons.csv       # add the dataset here
├── outputs/                  # generated results
├── src/
│   ├── analysis.py
│   ├── data.py
│   ├── database.py
│   ├── modeling.py
│   ├── run_analysis.py
│   └── visuals.py
├── tests/
│   └── test_project.py
├── requirements.txt
└── run.py
```

`app.py` is the main Streamlit entry point. The `app/dashboard.py` file is kept as a small alternate module for compatibility with the earlier project layout.

## Setup

Use Python 3.10 or newer.

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the analysis

Put the NBA season dataset in:

```text
data/all_seasons.csv
```

Then run:

```bash
python run.py
```

This creates the SQLite database, analysis tables, model files, metrics, predictions, and charts under `outputs/`.

## Run the dashboard

From the project root:

```bash
streamlit run app.py
```

If `data/all_seasons.csv` is not present, the dashboard uses a small clearly labeled demo dataset so the interface can still be tested. The demo data is not intended to represent real NBA results.

## Run the tests

```bash
pytest
```

## Data

The project was designed around publicly available NBA player season data. The expected CSV contains season-level player statistics including points, rebounds, assists, games played, efficiency metrics, draft information, and basic player information.

The dataset itself is not included in the repository. Add your own copy as `data/all_seasons.csv`.

## Limitations

- Player names are used as the main historical identifier, so traded players and name changes are not handled perfectly.
- The project uses season-level statistics rather than play-by-play information.
- A 15% improvement is a project-defined target, not an official NBA definition of a breakout.
- Random Forest models can identify relationships in historical data but do not establish causal reasons for player development.
- The model does not directly account for injuries, coaching changes, roster changes, or future playing-time decisions.

## What I learned

The main lesson from this project was that the modeling step is only one part of the workflow. The way the data is cleaned, organized, queried, and split has a large effect on the quality of the results.

It also gave me practice moving between SQL, Python analysis, machine learning, and visualization in one project instead of treating them as separate exercises.
