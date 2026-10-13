"""Prediction utilities for the Chennai heatwave proxy warning prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from src.feature_engineering import build_features
from src.train_model import TrainingResult


@dataclass(frozen=True)
class WarningPrediction:
    """Prediction for one target date.

    Class ``1`` is a heatwave proxy warning and is not an official IMD
    declaration. ``predicted_class_probability`` is the probability assigned
    to the predicted class, or ``None`` when the estimator has no
    ``predict_proba`` method.
    """

    target_date: pd.Timestamp
    predicted_class: int
    predicted_class_probability: Optional[float]


def predict_warning(
    training_result: TrainingResult,
    observations: pd.DataFrame,
    target_date: str | date | pd.Timestamp,
    *,
    date_column: str = "date",
    max_temperature_column: str = "max_temp_c",
    min_temperature_column: Optional[str] = "min_temp_c",
    humidity_column: Optional[str] = "humidity",
    wind_speed_column: Optional[str] = "wind_speed",
    lag_days: Sequence[int] = (1, 2, 3, 7),
    rolling_windows: Sequence[int] = (3, 7),
) -> WarningPrediction:
    """Predict the heatwave proxy class for one future target date.

    ``observations`` must contain daily historical rows plus exactly one row
    for ``target_date``. Historical weather should be available where
    observed. Target-day weather columns may be missing (``None``/``NaN``);
    they are not used as predictors. Departure-from-normal and label columns
    are not required and are never passed to the predictor model.

    The same :func:`build_features` settings used during training must be
    supplied here. Generated predictor names and their order are compared
    with ``training_result.feature_columns`` and, where available, the
    fitted estimator's ``feature_names_in_``. A mismatch raises ``ValueError``
    rather than reordering or silently altering the model schema.

    Rows after the target date are rejected. The date column is validated for
    invalid and duplicate calendar dates by the feature builder. Features use
    prior calendar-day weather plus target-date calendar features only;
    target-day Tmax, Tmin, humidity, wind, departure-from-normal, and labels
    are excluded from prediction inputs.

    Args:
        training_result: Successful result from
            :func:`src.train_model.train_baseline_model`.
        observations: Daily historical weather with one target-date row.
        target_date: The single date for which to produce a prediction.
        date_column: Date column name used by feature construction.
        max_temperature_column: Tmax column name used during training.
        min_temperature_column: Optional Tmin column name used during training.
        humidity_column: Optional humidity column name used during training.
        wind_speed_column: Optional wind-speed column name used during training.
        lag_days: The same calendar-day lags used during training.
        rolling_windows: The same rolling windows used during training.

    Returns:
        A :class:`WarningPrediction` with the normalized target date,
        predicted class, and probability of that predicted class when
        available.

    Raises:
        TypeError: If the input observations or training result has the wrong
            type.
        ValueError: If the target date is invalid or absent/not unique, there
            are dates later than the target, or the feature schema differs
            from the fitted training schema.
    """
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
        parsed_dates = pd.to_datetime(
            observations[date_column],
            errors="coerce",
            format="mixed",
        )
        normalized_dates = parsed_dates.dt.normalize()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Column {date_column!r} must contain consistently parseable dates."
        ) from exc

    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        invalid_rows = observations.index[invalid_dates].tolist()[:5]
        raise ValueError(
            f"Column {date_column!r} contains invalid or missing dates "
            f"at row index(es): {invalid_rows}."
        )
    duplicate_dates = normalized_dates.duplicated(keep=False)
    if duplicate_dates.any():
        raise ValueError(f"Duplicate daily dates in column {date_column!r}.")

    target_matches = normalized_dates.eq(normalized_target)
    if int(target_matches.sum()) != 1:
        raise ValueError(
            "Exactly one row matching target_date must be present in observations."
        )
    if normalized_dates.gt(normalized_target).any():
        raise ValueError("observations must not contain dates after target_date.")

    features = build_features(
        observations,
        date_column=date_column,
        max_temperature_column=max_temperature_column,
        min_temperature_column=min_temperature_column,
        humidity_column=humidity_column,
        wind_speed_column=wind_speed_column,
        lag_days=lag_days,
        rolling_windows=rolling_windows,
    )
    generated_columns = tuple(
        column for column in features.columns if column != date_column
    )
    expected_columns = tuple(training_result.feature_columns)
    if generated_columns != expected_columns:
        generated_set = set(generated_columns)
        expected_set = set(expected_columns)
        missing = [name for name in expected_columns if name not in generated_set]
        unexpected = [name for name in generated_columns if name not in expected_set]
        if missing or unexpected:
            raise ValueError(
                "Prediction feature schema does not match training. "
                f"Missing features: {missing}; unexpected features: {unexpected}."
            )
        raise ValueError(
            "Prediction feature order does not match the training feature order."
        )

    model = training_result.model
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and tuple(model_feature_names) != expected_columns:
        raise ValueError(
            "The fitted model feature order does not match "
            "training_result.feature_columns."
        )

    matching_feature_positions = features[date_column].eq(normalized_target).to_numpy()
    if int(matching_feature_positions.sum()) != 1:
        raise ValueError("Unable to uniquely align generated features to target_date.")
    model_input = features.loc[
        matching_feature_positions, list(expected_columns)
    ].iloc[[0]]

    predicted = model.predict(model_input)
    predicted_class = int(predicted[0])
    predicted_probability: Optional[float] = None
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if callable(predict_proba):
        if classes is None or predicted_class not in classes:
            raise ValueError(
                "The fitted model exposes predict_proba but no matching classes_."
            )
        class_position = list(classes).index(predicted_class)
        predicted_probability = float(predict_proba(model_input)[0][class_position])

    return WarningPrediction(
        target_date=normalized_target,
        predicted_class=predicted_class,
        predicted_class_probability=predicted_probability,
    )