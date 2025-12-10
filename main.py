"""
NBA Analytics Platform: Game Predictions and Fantasy Insights
CS 210 Project - FINAL VERSION
Andy Zhu, AZ455

You'll need the 'all_seasons.csv' dataset in the same folder.
You can grab it from Kaggle: NBA Player Stats (1950-2022).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import sqlite3
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')

class NBADatabase:
    def __init__(self, db_name='nba_analytics.db'):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        if os.path.exists(self.db_name):
            os.remove(self.db_name)
        self.conn = sqlite3.connect(self.db_name)
        print("Database connected")
        return self.conn

    def create_schema(self):
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                height REAL,
                weight REAL,
                college TEXT,
                country TEXT,
                draft_year INTEGER,
                draft_round INTEGER,
                draft_number INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_abbreviation TEXT NOT NULL,
                team_name TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE player_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                team_id INTEGER,
                season TEXT,
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
            )
        ''')

        cursor.execute('CREATE INDEX idx_player_stats_season ON player_stats(season)')
        cursor.execute('CREATE INDEX idx_player_stats_player_season ON player_stats(player_id, season)')

        self.conn.commit()
        print("Database schema created")

    def load_data(self, df):
        cursor = self.conn.cursor()

        unique_players = df[['player_name', 'player_height', 'player_weight',
                            'college', 'country', 'draft_year', 'draft_round',
                            'draft_number']].drop_duplicates('player_name')

        for _, row in unique_players.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO players 
                (player_name, height, weight, college, country, draft_year, draft_round, draft_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['player_name'], row['player_height'], row['player_weight'],
                  row['college'], row['country'], row['draft_year'],
                  row['draft_round'], row['draft_number']))

        unique_teams = df[['team_abbreviation']].drop_duplicates()
        for _, row in unique_teams.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO teams (team_abbreviation, team_name)
                VALUES (?, ?)
            ''', (row['team_abbreviation'], row['team_abbreviation']))

        inserted = 0
        for _, row in df.iterrows():
            cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (row['player_name'],))
            player_result = cursor.fetchone()
            if not player_result:
                continue
            player_id = player_result[0]

            cursor.execute('SELECT team_id FROM teams WHERE team_abbreviation = ?', (row['team_abbreviation'],))
            team_result = cursor.fetchone()
            if not team_result:
                continue
            team_id = team_result[0]

            cursor.execute('''
                INSERT INTO player_stats 
                (player_id, team_id, season, age, games_played, points, rebounds, 
                 assists, net_rating, oreb_pct, dreb_pct, usg_pct, ts_pct, ast_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (player_id, team_id, row['season'], row['age'], row['gp'],
                  row['pts'], row['reb'], row['ast'],
                  row['net_rating'], row['oreb_pct'], row['dreb_pct'],
                  row['usg_pct'], row['ts_pct'], row['ast_pct']))
            inserted += 1

        self.conn.commit()
        print(f"Loaded {inserted} stat records")

    def run_queries(self):
        print("\nSQL Queries Analysis")
        print("=" * 70)

        query1 = """
            SELECT 
                p.player_name,
                t.team_abbreviation,
                COUNT(DISTINCT ps.season) as seasons_played,
                ROUND(AVG(ps.points), 1) as career_ppg,
                ROUND(AVG(ps.rebounds), 1) as career_rpg,
                ROUND(AVG(ps.assists), 1) as career_apg,
                MAX(ps.points) as career_high_points,
                SUM(ps.games_played) as total_games
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.player_id
            JOIN teams t ON ps.team_id = t.team_id
            GROUP BY p.player_name, t.team_abbreviation
            HAVING seasons_played >= 3 AND career_ppg >= 15
            ORDER BY career_ppg DESC
            LIMIT 10
        """
        top_scorers = self.run_query(query1)
        print("\nTop 10 Scorers (15+ PPG, 3+ seasons):")
        print(top_scorers.to_string(index=False))

        query2 = """
            WITH player_seasons AS (
                SELECT 
                    p.player_name,
                    ps.season,
                    ps.age,
                    ps.points as ppg,
                    LAG(ps.points) OVER (PARTITION BY p.player_id ORDER BY ps.season) as prev_ppg,
                    LAG(ps.games_played) OVER (PARTITION BY p.player_id ORDER BY ps.season) as prev_gp
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE ps.games_played >= 20
            )
            SELECT 
                ROUND(AVG(CASE 
                    WHEN prev_ppg > 0 THEN 100.0 * (ppg - prev_ppg) / prev_ppg 
                    ELSE NULL 
                END), 1) as avg_improvement_pct,
                COUNT(*) as total_season_pairs
            FROM player_seasons
            WHERE prev_ppg IS NOT NULL AND prev_gp >= 20
        """
        improvement_analysis = self.run_query(query2)
        print(f"\nAverage year-over-year scoring improvement: {improvement_analysis['avg_improvement_pct'][0]}%")
        print(f"Based on {improvement_analysis['total_season_pairs'][0]:,} season pairs")

        return top_scorers, improvement_analysis

    def run_query(self, sql, params=None):
        return pd.read_sql_query(sql, self.conn, params=params)

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database closed")

class NBADataAnalyzer:
    def __init__(self):
        self.df = None
        self.num_cols = ['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast']

    def load_and_clean(self):
        print("Loading data...")
        self.df = pd.read_csv('all_seasons.csv')
        print(f"Loaded {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")

        if 'Unnamed: 0' in self.df.columns:
            self.df.drop('Unnamed: 0', axis=1, inplace=True)

        self.df['college'].fillna('Not Specified', inplace=True)
        self.df['country'].fillna('Not Specified', inplace=True)

        for col in ['draft_year', 'draft_round', 'draft_number']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        self.df['season_year'] = self.df['season'].str[:4].astype(int)

        print("Data cleaned")
        return self.df

    def explore_data(self):
        print("\nData Summary:")
        print(self.df[self.num_cols].describe())

        print(f"\nSeasons: {self.df['season'].min()} to {self.df['season'].max()}")

        self.make_temporal_charts()
        return self.df

    def make_temporal_charts(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        player_counts = self.df['player_name'].value_counts()
        multi_season_players = player_counts[player_counts >= 5].index[:20]

        axes[0, 0].set_title('Player Development Curves')
        axes[0, 0].set_xlabel('Season in Career')
        axes[0, 0].set_ylabel('Points Per Game')

        for player in multi_season_players[:10]:
            player_data = self.df[self.df['player_name'] == player].sort_values('season_year')
            seasons_in_career = range(1, len(player_data) + 1)
            axes[0, 0].plot(seasons_in_career, player_data['pts'], marker='o', label=player[:15])

        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        soph_jump = []
        for player in self.df['player_name'].unique():
            player_data = self.df[self.df['player_name'] == player].sort_values('season_year')
            if len(player_data) >= 2:
                rookie_stats = player_data.iloc[0]
                soph_stats = player_data.iloc[1]
                if rookie_stats['gp'] >= 20 and soph_stats['gp'] >= 20:
                    improvement = 100 * (soph_stats['pts'] - rookie_stats['pts']) / rookie_stats['pts']
                    soph_jump.append(improvement)

        axes[0, 1].hist(soph_jump, bins=30, color='orange', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('% Change in PPG (Rookie to 2nd Year)')
        axes[0, 1].set_ylabel('Number of Players')
        axes[0, 1].set_title('Sophomore Jump Analysis')
        axes[0, 1].grid(True, alpha=0.3)

        height_bins = [0, 190, 200, 210, 250]
        height_labels = ['Guard', 'Wing', 'Forward', 'Center']
        self.df['position_proxy'] = pd.cut(self.df['player_height'], bins=height_bins, labels=height_labels)

        for pos in height_labels:
            pos_data = self.df[self.df['position_proxy'] == pos]
            age_scoring = pos_data.groupby('age')['pts'].mean()
            axes[0, 2].plot(age_scoring.index, age_scoring.values, marker='o', label=pos)

        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Average Points')
        axes[0, 2].set_title('Scoring by Age and Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        draft_data = self.df[self.df['draft_number'] > 0].copy()
        draft_data['draft_tier'] = pd.cut(draft_data['draft_number'],
                                         bins=[0, 5, 15, 30, 60, 300],
                                         labels=['Top 5', 'Lottery', 'Late 1st', '2nd Rd', 'Undrafted'])

        career_ppg_by_draft = draft_data.groupby(['draft_tier', 'season_year'])['pts'].mean().unstack(0)
        axes[1, 0].plot(career_ppg_by_draft.index, career_ppg_by_draft.values, linewidth=2)
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Average PPG')
        axes[1, 0].set_title('Draft Position vs Career Scoring')
        axes[1, 0].legend(career_ppg_by_draft.columns)
        axes[1, 0].grid(True, alpha=0.3)

        self.df = self.df.sort_values(['player_name', 'season_year'])
        self.df['prev_pts'] = self.df.groupby('player_name')['pts'].shift(1)
        self.df['prev_gp'] = self.df.groupby('player_name')['gp'].shift(1)

        valid_pairs = self.df.dropna(subset=['prev_pts', 'pts'])
        axes[1, 1].scatter(valid_pairs['prev_pts'], valid_pairs['pts'], alpha=0.3, color='green')
        axes[1, 1].set_xlabel('Points in Season N-1')
        axes[1, 1].set_ylabel('Points in Season N')
        axes[1, 1].set_title('Year-over-Year Scoring Consistency')
        axes[1, 1].grid(True, alpha=0.3)

        z = np.polyfit(valid_pairs['prev_pts'], valid_pairs['pts'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(valid_pairs['prev_pts'], p(valid_pairs['prev_pts']), "r--", linewidth=2)
        axes[1, 1].text(5, 30, f'Correlation: {valid_pairs["prev_pts"].corr(valid_pairs["pts"]):.3f}',
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        gp_variance = self.df.groupby('player_name')['gp'].std() / self.df.groupby('player_name')['gp'].mean()
        axes[1, 2].hist(gp_variance.dropna(), bins=30, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Games Played Consistency')
        axes[1, 2].set_ylabel('Number of Players')
        axes[1, 2].set_title('Player Durability')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.df.drop(['prev_pts', 'prev_gp'], axis=1, inplace=True)

class NBAMachineLearning:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}

    def prepare_temporal_data(self, df):
        print("Preparing temporal data...")

        df = df.sort_values(['player_name', 'season_year']).copy()
        df = df[df['gp'] >= 20].copy()

        player_seasons = []

        for player_name, player_data in df.groupby('player_name'):
            player_data = player_data.sort_values('season_year')

            if len(player_data) < 2:
                continue

            for i in range(1, len(player_data)):
                current_season = player_data.iloc[i]
                prev_season = player_data.iloc[i-1]

                if prev_season['gp'] < 20 or current_season['gp'] < 20:
                    continue

                season_gap = current_season['season_year'] - prev_season['season_year']
                if season_gap > 1:
                    continue

                if i == 1:
                    career_ppg = prev_season['pts']
                    career_gp = prev_season['gp']
                else:
                    career_ppg = player_data.iloc[:i]['pts'].mean()
                    career_gp = player_data.iloc[:i]['gp'].mean()

                features = {
                    'player_name': player_name,
                    'target_season': current_season['season'],
                    'prev_season': prev_season['season'],

                    'height': prev_season['player_height'],
                    'weight': prev_season['player_weight'],

                    'age': prev_season['age'],
                    'age_squared': prev_season['age'] ** 2,
                    'seasons_in_league': i,
                    'years_since_draft': max(prev_season['season_year'] - prev_season['draft_year'], 0),

                    'prev_ppg': prev_season['pts'],
                    'prev_rpg': prev_season['reb'],
                    'prev_apg': prev_season['ast'],
                    'prev_gp': prev_season['gp'],
                    'prev_ts_pct': prev_season['ts_pct'],
                    'prev_usg_pct': prev_season['usg_pct'],
                    'prev_net_rating': prev_season['net_rating'],
                    'prev_oreb_pct': prev_season['oreb_pct'],
                    'prev_dreb_pct': prev_season['dreb_pct'],
                    'prev_ast_pct': prev_season['ast_pct'],

                    'career_ppg': career_ppg,
                    'career_gp': career_gp,

                    'draft_round': max(prev_season['draft_round'], 1),
                    'draft_number': prev_season['draft_number'] if prev_season['draft_number'] > 0 else 60,
                    'draft_quality': 4 if prev_season['draft_number'] <= 5 else
                                   3 if prev_season['draft_number'] <= 15 else
                                   2 if prev_season['draft_number'] <= 30 else
                                   1 if prev_season['draft_number'] <= 60 else 0,

                    'height_category': 0 if prev_season['player_height'] < 190 else
                                    1 if prev_season['player_height'] < 200 else
                                    2 if prev_season['player_height'] < 210 else 3,

                    'next_ppg': current_season['pts'],
                    'next_rpg': current_season['reb'],
                    'next_apg': current_season['ast'],
                    'next_gp': current_season['gp'],
                    'next_fantasy_points': current_season['pts'] + current_season['reb'] * 1.2 + current_season['ast'] * 1.5,

                    'was_rookie': 1 if i == 1 else 0,
                }

                player_seasons.append(features)

        temporal_df = pd.DataFrame(player_seasons)

        if len(temporal_df) == 0:
            print("Error: No temporal data created")
            return None, None

        temporal_df['ppg_improvement_pct'] = 100 * (temporal_df['next_ppg'] - temporal_df['prev_ppg']) / temporal_df['prev_ppg'].clip(lower=0.1)
        temporal_df['will_improve_15pct'] = (temporal_df['ppg_improvement_pct'] > 15).astype(int)

        scoring_threshold = temporal_df['next_ppg'].quantile(0.85)
        temporal_df['next_high_scorer'] = (temporal_df['next_ppg'] > scoring_threshold).astype(int)

        print(f"Created {len(temporal_df)} season transitions")
        print(f"High scorer threshold: {scoring_threshold:.1f} PPG")
        print(f"Data covers: {temporal_df['target_season'].min()} to {temporal_df['target_season'].max()}")

        return temporal_df, scoring_threshold

    def temporal_train_test_split(self, df, test_years=[2021, 2022], val_years=[2019, 2020]):
        df = df.copy()

        if 'target_season' not in df.columns:
            print("Error: target_season column not found")
            return None

        df['target_year'] = df['target_season'].str[:4].astype(int)

        train_mask = df['target_year'] < min(val_years)
        X_train = df[train_mask].drop(columns=['next_ppg', 'next_fantasy_points',
                                              'will_improve_15pct', 'next_high_scorer',
                                              'player_name', 'target_season', 'prev_season',
                                              'ppg_improvement_pct', 'next_rpg', 'next_apg', 'next_gp',
                                              'target_year'], errors='ignore')
        y_train_reg = df[train_mask]['next_fantasy_points']
        y_train_cls = df[train_mask]['next_high_scorer']

        val_mask = df['target_year'].isin(val_years)
        X_val = df[val_mask].drop(columns=['next_ppg', 'next_fantasy_points',
                                          'will_improve_15pct', 'next_high_scorer',
                                          'player_name', 'target_season', 'prev_season',
                                          'ppg_improvement_pct', 'next_rpg', 'next_apg', 'next_gp',
                                          'target_year'], errors='ignore')
        y_val_reg = df[val_mask]['next_fantasy_points']
        y_val_cls = df[val_mask]['next_high_scorer']

        test_mask = df['target_year'].isin(test_years)
        X_test = df[test_mask].drop(columns=['next_ppg', 'next_fantasy_points',
                                            'will_improve_15pct', 'next_high_scorer',
                                            'player_name', 'target_season', 'prev_season',
                                            'ppg_improvement_pct', 'next_rpg', 'next_apg', 'next_gp',
                                            'target_year'], errors='ignore')
        y_test_reg = df[test_mask]['next_fantasy_points']
        y_test_cls = df[test_mask]['next_high_scorer']

        print(f"\nTemporal Split:")
        print(f"Training: {train_mask.sum():,} samples ({df[train_mask]['target_year'].min()}-{df[train_mask]['target_year'].max()})")
        print(f"Validation: {val_mask.sum():,} samples ({df[val_mask]['target_year'].min()}-{df[val_mask]['target_year'].max()})")
        print(f"Test: {test_mask.sum():,} samples ({df[test_mask]['target_year'].min()}-{df[test_mask]['target_year'].max()})")

        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

        return (X_train, X_val, X_test,
                y_train_reg, y_val_reg, y_test_reg,
                y_train_cls, y_val_cls, y_test_cls)

    def train_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nTraining High Scorer Classifier...")

        self.scalers['classifier'] = StandardScaler()
        X_train_scaled = self.scalers['classifier'].fit_transform(X_train)
        X_val_scaled = self.scalers['classifier'].transform(X_val)
        X_test_scaled = self.scalers['classifier'].transform(X_test)

        self.models['classifier'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_leaf=5
        )

        self.models['classifier'].fit(X_train_scaled, y_train)

        y_val_pred = self.models['classifier'].predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        y_test_pred = self.models['classifier'].predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Validation Accuracy: {val_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=['Regular Scorer', 'High Scorer']))

        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=['Regular', 'High'])
        disp.plot(cmap='Blues')
        plt.title('High Scorer Prediction')
        plt.show()

        print("\nTop 10 Features:")
        feat_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.models['classifier'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        print(feat_imp.to_string(index=False))

        self.results['classifier'] = {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'features': X_train.columns.tolist()
        }

        return test_accuracy

    def train_regressor(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nTraining Fantasy Points Regressor...")

        self.models['regressor'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_leaf=5
        )

        self.models['regressor'].fit(X_train, y_train)

        prev_fantasy = X_test['prev_ppg'] + X_test['prev_rpg'] * 1.2 + X_test['prev_apg'] * 1.5
        baseline_mae = mean_absolute_error(y_test, prev_fantasy)
        baseline_r2 = r2_score(y_test, prev_fantasy)

        y_val_pred = self.models['regressor'].predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        y_test_pred = self.models['regressor'].predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        print(f"\nBaseline (Previous Season): R² = {baseline_r2:.3f}, MAE = {baseline_mae:.1f}")
        print(f"Validation: R² = {val_r2:.3f}, MAE = {val_mae:.1f}")
        print(f"Test: R² = {test_r2:.3f}, MAE = {test_mae:.1f}")
        print(f"Improvement: {test_r2 - baseline_r2:+.3f} R²")

        print("\nTop 10 Features:")
        feat_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.models['regressor'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        print(feat_imp.to_string(index=False))

        self.plot_regression_results(y_test, y_test_pred, prev_fantasy)

        self.results['regressor'] = {
            'baseline_r2': baseline_r2,
            'baseline_mae': baseline_mae,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'features': X_train.columns.tolist()
        }

        return test_r2, test_mae

    def plot_regression_results(self, y_true, y_pred, y_baseline):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, color='blue', label='Model Predictions')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Fantasy Points')
        axes[0, 0].set_ylabel('Predicted Fantasy Points')
        axes[0, 0].set_title('Model: Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(y_true, y_baseline, alpha=0.5, color='green', label='Baseline (Prev Season)')
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Fantasy Points')
        axes[0, 1].set_ylabel('Baseline Prediction')
        axes[0, 1].set_title('Baseline: Actual vs Predicted')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='purple')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Fantasy Points')
        axes[1, 0].set_ylabel('Residual (Actual - Predicted)')
        axes[1, 0].set_title('Model Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def evaluate_by_player_type(self, temporal_df, X_test, y_test_reg, y_test_cls, y_test_pred_reg, y_test_pred_cls):
        print("\nPerformance by Player Type")

        test_indices = X_test.index
        test_data = temporal_df.loc[test_indices].copy()
        test_data['pred_fantasy'] = y_test_pred_reg
        test_data['pred_high_scorer'] = y_test_pred_cls
        test_data['actual_fantasy'] = y_test_reg.values
        test_data['actual_high_scorer'] = y_test_cls.values

        test_data['player_type'] = 'Other'
        test_data.loc[test_data['seasons_in_league'] <= 2, 'player_type'] = 'Young (≤2 years)'
        test_data.loc[test_data['seasons_in_league'].between(3, 6), 'player_type'] = 'Prime (3-6 years)'
        test_data.loc[test_data['seasons_in_league'] >= 7, 'player_type'] = 'Veteran (≥7 years)'
        test_data.loc[test_data['prev_ppg'] >= 20, 'player_type'] = 'Star (20+ PPG)'
        test_data.loc[test_data['draft_quality'] == 4, 'player_type'] = 'Top Pick (Top 5)'

        for ptype in test_data['player_type'].unique():
            subset = test_data[test_data['player_type'] == ptype]
            if len(subset) < 10:
                continue

            mae = mean_absolute_error(subset['actual_fantasy'], subset['pred_fantasy'])
            r2 = r2_score(subset['actual_fantasy'], subset['pred_fantasy'])
            accuracy = accuracy_score(subset['actual_high_scorer'], subset['pred_high_scorer'])

            print(f"\n{ptype} ({len(subset)} players):")
            print(f"  Fantasy MAE: {mae:.1f}, R²: {r2:.3f}")
            print(f"  High Scorer Accuracy: {accuracy:.3f}")

    def test_on_real_players(self, temporal_df):
        print("\nReal Player Predictions (2021-22 Season)")

        test_2022 = temporal_df[temporal_df['target_season'].str.contains('2022')].copy()

        if len(test_2022) == 0:
            print("No 2022 data found. Using most recent available.")
            test_2022 = temporal_df.iloc[-100:].copy()

        X_real = test_2022.drop(columns=['next_ppg', 'next_fantasy_points',
                                         'will_improve_15pct', 'next_high_scorer',
                                         'player_name', 'target_season', 'prev_season',
                                         'ppg_improvement_pct', 'next_rpg', 'next_apg', 'next_gp',
                                         'target_year'], errors='ignore')

        for col in X_real.columns:
            if X_real[col].isnull().any():
                X_real[col].fillna(X_real[col].median(), inplace=True)

        if 'classifier' in self.models:
            X_real_scaled = self.scalers['classifier'].transform(X_real)
            high_scorer_probs = self.models['classifier'].predict_proba(X_real_scaled)[:, 1]
            high_scorer_preds = self.models['classifier'].predict(X_real_scaled)

        if 'regressor' in self.models:
            fantasy_preds = self.models['regressor'].predict(X_real)

        test_2022['pred_fantasy'] = fantasy_preds if 'regressor' in self.models else 0
        test_2022['pred_high_scorer_prob'] = high_scorer_probs if 'classifier' in self.models else 0
        test_2022['pred_high_scorer'] = high_scorer_preds if 'classifier' in self.models else 0

        target_players = ['LeBron James', 'Stephen Curry', 'Giannis Antetokounmpo',
                          'Nikola Jokic', 'Ja Morant', 'Trae Young']

        print("\nCase Studies:")
        for player in target_players:
            player_data = test_2022[test_2022['player_name'] == player]
            if len(player_data) > 0:
                row = player_data.iloc[0]
                print(f"\n{player}:")
                print(f"  2020-21: {row['prev_ppg']:.1f} PPG")
                print(f"  Predicted 2021-22: {row['pred_fantasy']:.1f} fantasy pts")
                print(f"  Actual 2021-22: {row['next_fantasy_points']:.1f} fantasy pts")
                print(f"  Error: {row['pred_fantasy'] - row['next_fantasy_points']:+.1f} pts")
                print(f"  High scorer prediction: {'Correct' if row['pred_high_scorer'] == row['next_high_scorer'] else 'Incorrect'}")

        test_2022['fantasy_error'] = test_2022['pred_fantasy'] - test_2022['next_fantasy_points']
        test_2022['abs_fantasy_error'] = test_2022['fantasy_error'].abs()
        test_2022['classification_correct'] = test_2022['pred_high_scorer'] == test_2022['next_high_scorer']

        print("\nBest Predictions:")
        best = test_2022.nsmallest(5, 'abs_fantasy_error')[['player_name', 'prev_ppg', 'next_ppg',
                                                            'pred_fantasy', 'next_fantasy_points',
                                                            'fantasy_error']]
        print(best.to_string(index=False))

        print("\nWorst Predictions:")
        worst = test_2022.nlargest(5, 'abs_fantasy_error')[['player_name', 'prev_ppg', 'next_ppg',
                                                            'pred_fantasy', 'next_fantasy_points',
                                                            'fantasy_error']]
        print(worst.to_string(index=False))

        high_scorer_threshold = test_2022['next_ppg'].quantile(0.85)
        predicted_breakouts = test_2022[(test_2022['pred_high_scorer'] == 1) &
                                        (test_2022['prev_ppg'] < high_scorer_threshold)]

        if len(predicted_breakouts) > 0:
            print(f"\nPredicted Breakouts (> {high_scorer_threshold:.1f} PPG):")
            for _, row in predicted_breakouts.head(5).iterrows():
                improvement = row['next_ppg'] - row['prev_ppg']
                predicted_correctly = row['next_high_scorer'] == 1
                status = "✓" if predicted_correctly else "✗"
                print(f"  {row['player_name']}: {row['prev_ppg']:.1f} → {row['next_ppg']:.1f} PPG ({improvement:+.1f}) {status}")

        actual_breakouts = test_2022[(test_2022['next_high_scorer'] == 1) &
                                     (test_2022['pred_high_scorer'] == 0)]

        if len(actual_breakouts) > 0:
            print(f"\nUnexpected Breakouts:")
            for _, row in actual_breakouts.head(5).iterrows():
                improvement = row['next_ppg'] - row['prev_ppg']
                print(f"  {row['player_name']}: {row['prev_ppg']:.1f} → {row['next_ppg']:.1f} PPG ({improvement:+.1f})")

        classification_accuracy = test_2022['classification_correct'].mean()
        avg_abs_error = test_2022['abs_fantasy_error'].mean()

        print(f"\nSummary:")
        print(f"  Classification Accuracy: {classification_accuracy:.1%}")
        print(f"  Average Absolute Error: {avg_abs_error:.1f} fantasy points")
        print(f"  Total Predictions: {len(test_2022)} players")

        synthetic_cases = [
            {
                'name': 'Young Breakout Candidate',
                'features': {
                    'age': 21,
                    'seasons_in_league': 2,
                    'was_rookie': 1,
                    'prev_ppg': 14.5,
                    'prev_rpg': 4.2,
                    'prev_apg': 3.8,
                    'prev_gp': 72,
                    'prev_ts_pct': 0.55,
                    'prev_usg_pct': 22.5,
                    'draft_quality': 3,
                    'height': 198,
                    'weight': 95,
                    'career_ppg': 14.5,
                    'career_gp': 72
                }
            },
            {
                'name': 'Aging Veteran',
                'features': {
                    'age': 35,
                    'seasons_in_league': 15,
                    'was_rookie': 0,
                    'prev_ppg': 12.3,
                    'prev_rpg': 3.5,
                    'prev_apg': 2.1,
                    'prev_gp': 58,
                    'prev_ts_pct': 0.52,
                    'prev_usg_pct': 18.5,
                    'draft_quality': 2,
                    'height': 203,
                    'weight': 107,
                    'career_ppg': 13.8,
                    'career_gp': 65
                }
            }
        ]

        print("\nHypothetical Scenarios:")
        for case in synthetic_cases:
            print(f"\n{case['name']}:")

            feature_vector = {}
            for feature in self.results['classifier']['features']:
                if feature in case['features']:
                    feature_vector[feature] = case['features'][feature]
                else:
                    if 'prev_' in feature and 'prev_net_rating' in feature:
                        feature_vector[feature] = 0
                    elif 'prev_' in feature and 'prev_oreb_pct' in feature:
                        feature_vector[feature] = 0.1
                    elif 'prev_' in feature and 'prev_dreb_pct' in feature:
                        feature_vector[feature] = 0.2
                    elif 'prev_' in feature and 'prev_ast_pct' in feature:
                        feature_vector[feature] = 20.0
                    elif feature == 'age_squared':
                        feature_vector[feature] = case['features']['age'] ** 2
                    elif feature == 'years_since_draft':
                        feature_vector[feature] = case['features']['seasons_in_league']
                    elif feature == 'height_category':
                        height = case['features']['height']
                        if height < 190:
                            feature_vector[feature] = 0
                        elif height < 200:
                            feature_vector[feature] = 1
                        elif height < 210:
                            feature_vector[feature] = 2
                        else:
                            feature_vector[feature] = 3
                    else:
                        feature_vector[feature] = 0

            X_test = pd.DataFrame([feature_vector])

            try:
                if 'classifier' in self.models:
                    X_test_scaled = self.scalers['classifier'].transform(X_test)
                    high_scorer_prob = self.models['classifier'].predict_proba(X_test_scaled)[0][1]
                    high_scorer_pred = self.models['classifier'].predict(X_test_scaled)[0]

                if 'regressor' in self.models:
                    fantasy_pred = self.models['regressor'].predict(X_test)[0]
                    print(f"  Predicted Fantasy Points: {fantasy_pred:.1f}")

                if 'classifier' in self.models:
                    print(f"  High Scorer Probability: {high_scorer_prob:.1%}")
                    print(f"  Prediction: {'High scorer' if high_scorer_pred == 1 else 'Regular scorer'}")

            except Exception as e:
                print(f"  Error: {str(e)[:50]}")

        return test_2022

    def run_comprehensive_analysis(self, df):
        print("\nRunning Comprehensive Analysis")

        temporal_df, scoring_threshold = self.prepare_temporal_data(df)

        if temporal_df is None:
            print("Failed to prepare temporal data")
            return

        splits = self.temporal_train_test_split(temporal_df)
        if splits is None:
            print("Failed to split data")
            return

        (X_train, X_val, X_test,
         y_train_reg, y_val_reg, y_test_reg,
         y_train_cls, y_val_cls, y_test_cls) = splits

        cls_acc = self.train_classifier(X_train, X_val, X_test,
                                       y_train_cls, y_val_cls, y_test_cls)

        reg_r2, reg_mae = self.train_regressor(X_train, X_val, X_test,
                                             y_train_reg, y_val_reg, y_test_reg)

        X_test_scaled = self.scalers['classifier'].transform(X_test) if 'classifier' in self.scalers else X_test
        y_test_pred_cls = self.models['classifier'].predict(X_test_scaled) if 'classifier' in self.models else None
        y_test_pred_reg = self.models['regressor'].predict(X_test) if 'regressor' in self.models else None

        self.evaluate_by_player_type(temporal_df, X_test, y_test_reg, y_test_cls,
                                   y_test_pred_reg, y_test_pred_cls)

        self.test_on_real_players(temporal_df)

        joblib.dump(self.models, 'nba_models.pkl')
        joblib.dump(self.scalers, 'nba_scalers.pkl')

        print("\nAnalysis Complete")
        print(f"  High Scorer Accuracy: {cls_acc:.3f}")
        print(f"  Fantasy Points R²: {reg_r2:.3f}")
        print(f"  Fantasy Points MAE: {reg_mae:.1f}")
        print("\nModels saved to 'nba_models.pkl' and 'nba_scalers.pkl'")

def main():
    print("NBA Analytics Platform")
    print("Andy Zhu, AZ455")
    print("="*50)

    db = NBADatabase()
    analyzer = NBADataAnalyzer()
    ml = NBAMachineLearning()

    try:
        print("\nPhase 1: Database Setup")
        db.connect()
        db.create_schema()

        df = analyzer.load_and_clean()
        db.load_data(df)
        db.run_queries()

        print("\nPhase 2: Data Exploration")
        analyzer.explore_data()

        print("\nPhase 3: Machine Learning")
        ml.run_comprehensive_analysis(df)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()

if __name__ == "__main__":
    main()