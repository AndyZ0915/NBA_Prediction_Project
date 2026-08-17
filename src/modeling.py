"""Simple temporal machine learning models for player development."""

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from .data import feature_columns


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class PlayerDevelopmentModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regressor = RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=random_state
        )
        self.classifier = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=4,
            class_weight="balanced", random_state=random_state
        )
        self.regression_metrics: dict = {}
        self.classification_metrics: dict = {}
        self.feature_names = feature_columns()
        self.feature_medians = pd.Series(dtype=float)

    @staticmethod
    def temporal_split(data: pd.DataFrame) -> SplitData:
        years = sorted(data["target_year"].unique())
        if len(years) < 5:
            raise ValueError("At least five target seasons are needed for a train/validation/test split.")

        test_years = years[-2:]
        validation_years = years[-4:-2]
        train_years = years[:-4]

        return SplitData(
            train=data[data.target_year.isin(train_years)].copy(),
            validation=data[data.target_year.isin(validation_years)].copy(),
            test=data[data.target_year.isin(test_years)].copy(),
        )

    def fit(self, transitions: pd.DataFrame) -> SplitData:
        split = self.temporal_split(transitions)
        x_train = split.train[self.feature_names].copy()
        x_val = split.validation[self.feature_names].copy()
        x_test = split.test[self.feature_names].copy()

        # Fill missing values using only training data so the validation and test
        # sets do not influence preprocessing.
        medians = x_train.median(numeric_only=True)
        self.feature_medians = medians
        x_train = x_train.fillna(medians)
        x_val = x_val.fillna(medians)
        x_test = x_test.fillna(medians)

        self.regressor.fit(x_train, split.train["next_ppg"])
        self.classifier.fit(x_train, split.train["improved_15pct"])

        val_reg = self.regressor.predict(x_val)
        test_reg = self.regressor.predict(x_test)
        val_cls = self.classifier.predict(x_val)
        test_cls = self.classifier.predict(x_test)
        test_prob = self.classifier.predict_proba(x_test)[:, 1]

        baseline = split.test["prev_ppg"].to_numpy()
        self.regression_metrics = {
            "validation_mae": mean_absolute_error(split.validation.next_ppg, val_reg),
            "validation_r2": r2_score(split.validation.next_ppg, val_reg),
            "test_mae": mean_absolute_error(split.test.next_ppg, test_reg),
            "test_r2": r2_score(split.test.next_ppg, test_reg),
            "baseline_mae": mean_absolute_error(split.test.next_ppg, baseline),
            "baseline_r2": r2_score(split.test.next_ppg, baseline),
        }

        roc_auc = (
            roc_auc_score(split.test.improved_15pct, test_prob)
            if split.test.improved_15pct.nunique() == 2
            else None
        )
        self.classification_metrics = {
            "validation_accuracy": accuracy_score(split.validation.improved_15pct, val_cls),
            "test_accuracy": accuracy_score(split.test.improved_15pct, test_cls),
            "test_precision": precision_score(split.test.improved_15pct, test_cls, zero_division=0),
            "test_recall": recall_score(split.test.improved_15pct, test_cls, zero_division=0),
            "test_f1": f1_score(split.test.improved_15pct, test_cls, zero_division=0),
            "test_roc_auc": roc_auc,
        }

        predictions = split.test[["player_name", "target_season", "prev_ppg", "next_ppg", "ppg_change_pct"]].copy()
        predictions["predicted_ppg"] = test_reg
        predictions["ppg_error"] = predictions["next_ppg"] - predictions["predicted_ppg"]
        predictions["improvement_probability"] = test_prob
        predictions["predicted_improvement"] = test_cls
        self.test_predictions = predictions.sort_values("improvement_probability", ascending=False)
        self.confusion = confusion_matrix(split.test.improved_15pct, test_cls, labels=[0, 1])
        self.classification_report = classification_report(
            split.test.improved_15pct, test_cls, labels=[0, 1],
            target_names=["No 15% improvement", "15%+ improvement"], zero_division=0, output_dict=True
        )
        return split

    def predict_player(self, row: pd.Series | dict) -> dict:
        features = pd.DataFrame([row])[self.feature_names]
        features = features.apply(pd.to_numeric, errors="coerce")
        if self.feature_medians.empty:
            features = features.fillna(0)
        else:
            features = features.fillna(self.feature_medians).fillna(0)
        ppg = float(self.regressor.predict(features)[0])
        probability = float(self.classifier.predict_proba(features)[0, 1])
        return {
            "predicted_ppg": ppg,
            "improvement_probability": probability,
            "likely_improvement": probability >= 0.5,
        }

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_names,
            "regression_importance": self.regressor.feature_importances_,
            "classification_importance": self.classifier.feature_importances_,
        }).sort_values("regression_importance", ascending=False)

    def save(self, directory: str) -> None:
        joblib.dump(self.regressor, f"{directory}/ppg_regressor.joblib")
        joblib.dump(self.classifier, f"{directory}/improvement_classifier.joblib")
