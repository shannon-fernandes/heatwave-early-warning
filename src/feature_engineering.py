"""Leakage-safe weather predictors for the dashboard/ML engine.

Predictors use only information available BEFORE the target date:
calendar-day lags, past-only Tmax rolling mean/max, and calendar terms.
Missing days or readings stay missing (nothing is fabricated); imputation, if
any, happens later inside the model pipeline, fitted on training rows only.

(The config-driven batch pipeline version lives in ``pipeline_features.py``.)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from src.validation import parse_daily_dates, require_numeric


def _positive_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    out = tuple(values)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in out):
        raise ValueError(f"{name} must contain positive integers.")
    return tuple(int(v) for v in out)


def build_features(
    observations: pd.DataFrame,
    *,
    date_column: str = "date",
    max_temperature_column: str = "max_temp_c",
    min_temperature_column: Optional[str] = "min_temp_c",
    humidity_column: Optional[str] = "humidity",
    wind_speed_column: Optional[str] = "wind_speed",
    departure_column: Optional[str] = None,
    lag_days: Sequence[int] = (1, 2, 3, 7),
    rolling_windows: Sequence[int] = (3, 7),
) -> pd.DataFrame:
    """Return one predictor row per observation, sorted by date.

    The result keeps each source row's original index label and has the date as its
    first column, so targets can be aligned by index. Optional weather columns are used
    only when present. ``departure_column`` (default None = off) adds LAGGED departure-from-normal
    predictors; only past days are used, never the target day. The input is not modified.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if observations.empty:
        raise ValueError("observations must contain at least one daily observation.")
    lags = _positive_ints(lag_days, "lag_days")
    windows = _positive_ints(rolling_windows, "rolling_windows")

    dates = parse_daily_dates(observations, date_column)
    weather = {max_temperature_column: require_numeric(observations, max_temperature_column)}
    for column in (min_temperature_column, humidity_column, wind_speed_column):
        if column is not None and column in observations.columns:
            weather[column] = require_numeric(observations, column)
    if departure_column is not None:                       # explicitly requested, so it must exist
        weather[departure_column] = require_numeric(observations, departure_column)

    order = np.argsort(dates.to_numpy(), kind="stable")
    sorted_dates = pd.DatetimeIndex(dates.to_numpy()[order])
    calendar = pd.date_range(sorted_dates[0], sorted_dates[-1], freq="D")
    # Complete daily calendar: a missing day appears as NaN, so lags and windows count
    # calendar days, not rows.
    full = pd.DataFrame(
        {name: pd.Series(values.to_numpy()[order], index=sorted_dates).reindex(calendar)
         for name, values in weather.items()}
    )

    features = pd.DataFrame(index=observations.index[order])
    features[date_column] = sorted_dates.to_numpy()
    for name in weather:
        for k in lags:
            features[f"{name}_lag_{k}"] = full[name].shift(k).reindex(sorted_dates).to_numpy()
    past_tmax = full[max_temperature_column].shift(1)       # excludes the target day itself
    for w in windows:
        rolled = past_tmax.rolling(w, min_periods=w)
        features[f"{max_temperature_column}_rolling_mean_{w}"] = rolled.mean().reindex(sorted_dates).to_numpy()
        features[f"{max_temperature_column}_rolling_max_{w}"] = rolled.max().reindex(sorted_dates).to_numpy()
    features["month"] = sorted_dates.month.to_numpy()
    features["day_of_year"] = sorted_dates.dayofyear.to_numpy()
    return features
