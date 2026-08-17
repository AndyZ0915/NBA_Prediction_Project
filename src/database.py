"""Small SQLite layer used for the project's SQL analysis."""

import sqlite3
from pathlib import Path
import pandas as pd


class NBADatabase:
    """Create and query a simple player, team, and season-stat schema."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        return self.connection

    def create_schema(self) -> None:
        if self.connection is None:
            raise RuntimeError("Call connect() before create_schema().")
        self.connection.executescript("""
            DROP TABLE IF EXISTS player_stats;
            DROP TABLE IF EXISTS players;
            DROP TABLE IF EXISTS teams;

            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT UNIQUE NOT NULL,
                height REAL,
                weight REAL,
                college TEXT,
                country TEXT,
                draft_year INTEGER,
                draft_round INTEGER,
                draft_number INTEGER
            );

            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                team_abbreviation TEXT UNIQUE NOT NULL
            );

            CREATE TABLE player_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                season TEXT NOT NULL,
                season_year INTEGER NOT NULL,
                age REAL,
                games_played INTEGER,
                points REAL,
                rebounds REAL,
                assists REAL,
                net_rating REAL,
                oreb_pct REAL,
                dreb_pct REAL,
                usg_pct REAL,
                ts_pct REAL,
                ast_pct REAL,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (team_id) REFERENCES teams(team_id)
            );

            CREATE INDEX idx_player_stats_player_season
            ON player_stats(player_id, season_year);
        """)
        self.connection.commit()

    def load_data(self, df: pd.DataFrame) -> None:
        if self.connection is None:
            raise RuntimeError("Call connect() before load_data().")

        players = df[[
            "player_name", "player_height", "player_weight", "college", "country",
            "draft_year", "draft_round", "draft_number"
        ]].drop_duplicates("player_name").reset_index(drop=True)
        players = players.rename(columns={"player_height": "height", "player_weight": "weight"})
        players.insert(0, "player_id", range(1, len(players) + 1))

        teams = df[["team_abbreviation"]].drop_duplicates().reset_index(drop=True)
        teams.insert(0, "team_id", range(1, len(teams) + 1))

        player_map = dict(zip(players.player_name, players.player_id))
        team_map = dict(zip(teams.team_abbreviation, teams.team_id))

        stats = df[[
            "player_name", "team_abbreviation", "season", "season_year", "age", "gp",
            "pts", "reb", "ast", "net_rating", "oreb_pct", "dreb_pct", "usg_pct",
            "ts_pct", "ast_pct"
        ]].copy()
        stats["player_id"] = stats["player_name"].map(player_map)
        stats["team_id"] = stats["team_abbreviation"].map(team_map)
        stats = stats.drop(columns=["player_name", "team_abbreviation"])
        stats = stats.rename(columns={"gp": "games_played", "pts": "points", "reb": "rebounds", "ast": "assists"})

        players.to_sql("players", self.connection, if_exists="append", index=False)
        teams.to_sql("teams", self.connection, if_exists="append", index=False)
        stats.to_sql("player_stats", self.connection, if_exists="append", index=False)
        self.connection.commit()

    def query(self, sql: str) -> pd.DataFrame:
        if self.connection is None:
            raise RuntimeError("Database is not connected.")
        return pd.read_sql_query(sql, self.connection)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def run_sql_examples(db: NBADatabase) -> dict[str, pd.DataFrame]:
    """Return a few readable SQL analyses for the dashboard and report."""
    top_players = db.query("""
        SELECT p.player_name,
               ROUND(AVG(s.points), 1) AS avg_ppg,
               ROUND(AVG(s.rebounds), 1) AS avg_rpg,
               ROUND(AVG(s.assists), 1) AS avg_apg,
               COUNT(*) AS seasons
        FROM player_stats s
        JOIN players p ON p.player_id = s.player_id
        GROUP BY p.player_name
        HAVING COUNT(*) >= 3
        ORDER BY avg_ppg DESC
        LIMIT 15;
    """)

    year_over_year = db.query("""
        WITH seasons AS (
            SELECT p.player_name, s.season_year, s.points,
                   LAG(s.points) OVER (
                       PARTITION BY s.player_id ORDER BY s.season_year
                   ) AS previous_points
            FROM player_stats s
            JOIN players p ON p.player_id = s.player_id
        )
        SELECT season_year,
               ROUND(AVG(points - previous_points), 2) AS avg_ppg_change,
               COUNT(*) AS player_pairs
        FROM seasons
        WHERE previous_points IS NOT NULL
        GROUP BY season_year
        ORDER BY season_year;
    """)

    team_summary = db.query("""
        SELECT t.team_abbreviation,
               ROUND(AVG(s.points), 1) AS avg_ppg,
               ROUND(AVG(s.net_rating), 2) AS avg_net_rating,
               COUNT(DISTINCT s.player_id) AS players
        FROM player_stats s
        JOIN teams t ON t.team_id = s.team_id
        GROUP BY t.team_abbreviation
        HAVING COUNT(*) >= 20
        ORDER BY avg_ppg DESC
        LIMIT 15;
    """)
    return {"top_players": top_players, "year_over_year": year_over_year, "team_summary": team_summary}
