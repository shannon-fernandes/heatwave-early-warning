from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import build_features
from src.heatwave_labels import assign_heatwave_proxy_labels
from src.warning_logic import choose_threshold


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    confusion_matrix: pd.DataFrame
    training_rows: int
    validation_rows: int
    unknown_training_labels: int
    unknown_validation_labels: int
    boundary_censored_labels: int
    validation_positive_cases: int
    validation_note: Optional[str]
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    feature_columns: tuple[str, ...]
    # Added for evaluation and warning logic (all default to None for backward compatibility).
    validation_dates: Optional[pd.DatetimeIndex] = None       # known-label validation rows only
    validation_true: Optional[np.ndarray] = None
    validation_probability: Optional[np.ndarray] = None       # P(class 1), same rows
    training_dates: Optional[pd.DatetimeIndex] = None         # known-label training rows only
    training_true: Optional[np.ndarray] = None
    decision_threshold: Optional[float] = None                # cost-based, see warning_logic
    departure_column: Optional[str] = None                    # set when lagged departures are predictors


def train_baseline_model(
    observations: pd.DataFrame, *, date_column: str = "date", max_temperature_column: str = "max_temp_c",
    departure_column: str = "departure_from_normal_c", label_column: str = "heatwave_proxy",
    min_temperature_column: Optional[str] = "min_temp_c", humidity_column: Optional[str] = "humidity",
    wind_speed_column: Optional[str] = "wind_speed", max_temperature_threshold_c: float = 37.0,
    departure_threshold_c: float = 4.5, minimum_consecutive_days: int = 2,
    lag_days: Sequence[int] = (1, 2, 3, 7), rolling_windows: Sequence[int] = (3, 7),
    train_fraction: float = 0.8, minimum_validation_positive_cases: int = 5,
    cost_missed: float = 10.0, cost_false_alarm: float = 1.0, use_departure_features: bool = False,
) -> TrainingResult:
    """Train and evaluate a simple heatwave-proxy classifier chronologically.

    Labels come from ``assign_heatwave_proxy_labels``; predictors from ``build_features``
    (calendar-day lags, past-only Tmax rolling summaries, calendar terms), never same-day
    weather. The first ``train_fraction`` of rows (by date) train a median-imputation,
    standardisation and class-balanced Logistic Regression pipeline; the rest validate.
    Unknown labels are excluded, never treated as negatives, and a persistence run that
    straddles the split is censored on both sides. ``precision/recall/f1`` use the 0.5
    default decision rule; the returned ``decision_threshold`` is the cost-minimising
    probability threshold chosen on the FIRST HALF of the validation rows only, so the
    second half stays untouched for honest evaluation (see ``src/run_evaluation.py``).
    With ``use_departure_features=True`` lagged departure-from-normal values (previous days only)
    are added as predictors; prediction then needs that column for the history rows.
    The model is returned in memory; nothing is written to disk.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if (isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float))
            or not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be a finite number strictly between 0 and 1.")
    if (isinstance(minimum_validation_positive_cases, bool)
            or not isinstance(minimum_validation_positive_cases, int) or minimum_validation_positive_cases < 1):
        raise ValueError("minimum_validation_positive_cases must be a positive integer.")
    source = observations.copy().reset_index(drop=True)
    labelled = assign_heatwave_proxy_labels(
        source, date_column=date_column, max_temperature_column=max_temperature_column,
        departure_column=departure_column, label_column=label_column,
        max_temperature_threshold_c=max_temperature_threshold_c, departure_threshold_c=departure_threshold_c,
        minimum_consecutive_days=minimum_consecutive_days)
    features = build_features(
        source, date_column=date_column, max_temperature_column=max_temperature_column,
        min_temperature_column=min_temperature_column, humidity_column=humidity_column,
        wind_speed_column=wind_speed_column,
        departure_column=departure_column if use_departure_features else None,
        lag_days=lag_days, rolling_windows=rolling_windows)
    targets = labelled.loc[features.index, label_column]
    feature_columns = tuple(column for column in features.columns if column != date_column)
    predictors = features.loc[:, feature_columns]
    split_at = int(len(features) * train_fraction)
    if split_at < 1 or split_at >= len(features):
        raise ValueError("Not enough observations for the requested chronological train/validation split.")
    train_labels = targets.iloc[:split_at]
    validation_labels = targets.iloc[split_at:]
    boundary_censored_labels = 0
    if minimum_consecutive_days > 1:
        ordered_source = source.loc[features.index]
        maximum = pd.to_numeric(ordered_source[max_temperature_column], errors="raise")
        departure = pd.to_numeric(ordered_source[departure_column], errors="raise")
        known_conditions = maximum.notna() & departure.notna()
        qualifying = (maximum.ge(max_temperature_threshold_c) & departure.ge(departure_threshold_c)
                      & known_conditions).to_numpy(dtype=bool)
        ordered_dates = pd.DatetimeIndex(features[date_column])
        qualifying_run: list[int] = []

        def censor_split_spanning_run(run: list[int]) -> None:
            nonlocal boundary_censored_labels
            if run and run[0] < split_at <= run[-1]:
                targets.iloc[run] = pd.NA
                boundary_censored_labels += len(run)

        for position in range(len(features)):
            if not qualifying[position]:
                censor_split_spanning_run(qualifying_run)
                qualifying_run = []
                continue
            if (qualifying_run and ordered_dates[position] - ordered_dates[position - 1] != pd.Timedelta(days=1)):
                censor_split_spanning_run(qualifying_run)
                qualifying_run = []
            qualifying_run.append(position)
        censor_split_spanning_run(qualifying_run)
        train_labels = targets.iloc[:split_at]
        validation_labels = targets.iloc[split_at:]
    train_known = train_labels.notna()
    validation_known = validation_labels.notna()
    unknown_training_labels = int((~train_known).sum())
    unknown_validation_labels = int((~validation_known).sum())
    if not train_known.any():
        raise ValueError("No known labels remain in the chronological training partition.")
    if not validation_known.any():
        raise ValueError("No known labels remain in the chronological validation partition.")
    x_train = predictors.iloc[:split_at].loc[train_known]
    y_train = train_labels.loc[train_known].astype("int8")
    x_validation = predictors.iloc[split_at:].loc[validation_known]
    y_validation = validation_labels.loc[validation_known].astype("int8")
    if y_train.nunique() < 2:
        raise ValueError("The training partition must contain both negative and positive known labels to fit a binary classifier.")
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    metrics = {"precision": float(precision_score(y_validation, predictions, zero_division=0)),
               "recall": float(recall_score(y_validation, predictions, zero_division=0)),
               "f1": float(f1_score(y_validation, predictions, zero_division=0))}
    matrix = confusion_matrix(y_validation, predictions, labels=[0, 1])
    confusion = pd.DataFrame(matrix, index=pd.Index([0, 1], name="actual"), columns=pd.Index([0, 1], name="predicted"))
    validation_positive_cases = int(y_validation.sum())
    validation_note = None
    if validation_positive_cases < minimum_validation_positive_cases:
        validation_note = ("Validation metrics may be unreliable because the validation "
                           f"partition has only {validation_positive_cases} positive case(s); "
                           f"at least {minimum_validation_positive_cases} were requested for "
                           "a more informative positive-case evaluation.")
    validation_probability = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]
    half = max(1, len(y_validation) // 2)
    decision_threshold = choose_threshold(
        y_validation.to_numpy()[:half], validation_probability[:half], cost_missed, cost_false_alarm
    )["threshold"]
    train_dates = features[date_column].iloc[:split_at]
    validation_dates = features[date_column].iloc[split_at:]
    return TrainingResult(
        model=model, metrics=metrics, confusion_matrix=confusion, training_rows=len(y_train),
        validation_rows=len(y_validation), unknown_training_labels=unknown_training_labels,
        unknown_validation_labels=unknown_validation_labels, boundary_censored_labels=boundary_censored_labels,
        validation_positive_cases=validation_positive_cases, validation_note=validation_note,
        training_start=pd.Timestamp(train_dates.iloc[0]), training_end=pd.Timestamp(train_dates.iloc[-1]),
        validation_start=pd.Timestamp(validation_dates.iloc[0]), validation_end=pd.Timestamp(validation_dates.iloc[-1]),
        feature_columns=feature_columns,
        validation_dates=pd.DatetimeIndex(features.loc[x_validation.index, date_column]),
        validation_true=y_validation.to_numpy(), validation_probability=validation_probability,
        training_dates=pd.DatetimeIndex(features.loc[x_train.index, date_column]),
        training_true=y_train.to_numpy(), decision_threshold=float(decision_threshold),
        departure_column=departure_column if use_departure_features else None)
