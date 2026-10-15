from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.feature_engineering import build_features
from src.train_model import TrainingResult


@dataclass(frozen=True)
class WarningPrediction:
    target_date: pd.Timestamp
    predicted_class: int
    predicted_class_probability: Optional[float]
    # Added: probability of class 1 (heatwave proxy) and the largest drivers of it, as
    # (feature, standardised-coefficient contribution) pairs. Both are optional extras.
    positive_class_probability: Optional[float] = None
    contributing_factors: tuple = ()


def _contributing_factors(model, model_input: pd.DataFrame, top: int = 5) -> tuple:
    """Largest positive contributions (coefficient x standardised value) of a fitted
    imputer-scaler-logistic-regression pipeline; empty for any other estimator."""
    steps = getattr(model, "named_steps", None)
    if not isinstance(model, Pipeline) or not steps or not {"imputer", "scaler", "classifier"} <= set(steps):
        return ()
    coef = getattr(steps["classifier"], "coef_", None)
    if coef is None:
        return ()
    z = steps["scaler"].transform(steps["imputer"].transform(model_input))[0]
    contribution = coef[0] * z
    order = np.argsort(-np.abs(contribution))[:top]          # largest effects in either direction
    return tuple((str(model_input.columns[i]), float(contribution[i])) for i in order)


def predict_warning(training_result: TrainingResult, observations: pd.DataFrame,
                    target_date: str | date | pd.Timestamp, *, date_column: str = "date",
                    max_temperature_column: str = "max_temp_c", min_temperature_column: Optional[str] = "min_temp_c",
                    humidity_column: Optional[str] = "humidity", wind_speed_column: Optional[str] = "wind_speed",
                    lag_days: Sequence[int] = (1, 2, 3, 7), rolling_windows: Sequence[int] = (3, 7)) -> WarningPrediction:
    if not isinstance(training_result, TrainingResult):
        raise TypeError("training_result must be a TrainingResult from training.")
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if observations.empty:
        raise ValueError("observations must contain historical data and a target-date row.")
    try:
        parsed_target = pd.to_datetime(target_date, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError("target_date must be a valid calendar date.") from exc
    if pd.isna(parsed_target):
        raise ValueError("target_date must be a valid calendar date.")
    normalized_target = pd.Timestamp(parsed_target).normalize()
    if date_column not in observations.columns:
        raise ValueError(f"Missing required date column: {date_column!r}.")
    try:
        parsed_dates = pd.to_datetime(observations[date_column], errors="coerce", format="mixed")
        normalized_dates = parsed_dates.dt.normalize()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Column {date_column!r} must contain consistently parseable dates.") from exc
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        invalid_rows = observations.index[invalid_dates].tolist()[:5]
        raise ValueError(f"Column {date_column!r} contains invalid or missing dates at row index(es): {invalid_rows}.")
    if normalized_dates.duplicated(keep=False).any():
        raise ValueError(f"Duplicate daily dates in column {date_column!r}.")
    target_matches = normalized_dates.eq(normalized_target)
    if int(target_matches.sum()) != 1:
        raise ValueError("Exactly one row matching target_date must be present in observations.")
    if normalized_dates.gt(normalized_target).any():
        raise ValueError("observations must not contain dates after target_date.")
    features = build_features(observations, date_column=date_column, max_temperature_column=max_temperature_column,
                              min_temperature_column=min_temperature_column, humidity_column=humidity_column,
                              wind_speed_column=wind_speed_column,
                              departure_column=getattr(training_result, "departure_column", None),
                              lag_days=lag_days, rolling_windows=rolling_windows)
    generated_columns = tuple(column for column in features.columns if column != date_column)
    expected_columns = tuple(training_result.feature_columns)
    if generated_columns != expected_columns:
        generated_set, expected_set = set(generated_columns), set(expected_columns)
        missing = [n for n in expected_columns if n not in generated_set]
        unexpected = [n for n in generated_columns if n not in expected_set]
        if missing or unexpected:
            raise ValueError("Prediction feature schema does not match training. "
                             f"Missing features: {missing}; unexpected features: {unexpected}.")
        raise ValueError("Prediction feature order does not match the training feature order.")
    model = training_result.model
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and tuple(model_feature_names) != expected_columns:
        raise ValueError("The fitted model feature order does not match training_result.feature_columns.")
    pos = features[date_column].eq(normalized_target).to_numpy()
    if int(pos.sum()) != 1:
        raise ValueError("Unable to uniquely align generated features to target_date.")
    model_input = features.loc[pos, list(expected_columns)].iloc[[0]]
    predicted = model.predict(model_input)
    predicted_class = int(predicted[0])
    predicted_probability: Optional[float] = None
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if callable(predict_proba):
        if classes is None or predicted_class not in classes:
            raise ValueError("The fitted model exposes predict_proba but no matching classes_.")
        predicted_probability = float(predict_proba(model_input)[0][list(classes).index(predicted_class)])
    positive_probability: Optional[float] = None
    if callable(predict_proba) and classes is not None and 1 in list(classes):
        positive_probability = float(predict_proba(model_input)[0][list(classes).index(1)])
    return WarningPrediction(target_date=normalized_target, predicted_class=predicted_class,
                             predicted_class_probability=predicted_probability,
                             positive_class_probability=positive_probability,
                             contributing_factors=_contributing_factors(model, model_input))
