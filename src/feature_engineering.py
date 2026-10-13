"""Leakage-safe predictor construction for daily heatwave observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import pandas as pd


def build_features(
    observations: pd.DataFrame,
    *,
    date_column: str = "date",
    max_temperature_column: str = "max_temp_c",
    min_temperature_column: Optional[str] = "min_temp_c",
    humidity_column: Optional[str] = "humidity",
    wind_speed_column: Optional[str] = "wind_speed",
    lag_days: Sequence[int] = (1, 2, 3, 7),
    rolling_windows: Sequence[int] = (3, 7),
) -> pd.DataFrame:
    """Build date-aligned, past-only predictors from daily weather observations.

    Required columns are the date and daily maximum temperature. Minimum
    temperature, humidity, and wind speed are included when their configured
    columns are present; pass ``None`` for an optional column to disable it.
    All weather columns are numeric and used as potential lag predictors.

    Lags are exact prior calendar days: ``max_temp_c_lag_1`` is Tmax from the
    previous calendar day, and it is missing if that day's observation is
    absent. Rolling mean and maximum features use the preceding calendar-day
    window, excluding the target date, and are emitted only when every day in
    that window has a non-missing Tmax. Missing dates and measurements are not
    filled or imputed.

    The result contains only the normalized date column and engineered
    predictors. It intentionally excludes all other input columns, including
    target labels, so targets remain separate. Rows are sorted chronologically
    and retain their original index labels; the date column and index together
    provide a stable mapping back to source observations.

    Calendar features ``month`` and ``day_of_year`` use the target row's date,
    which is known at prediction time. Weather lags and rolling statistics do
    not use target-day weather observations.

    Args:
        observations: Daily weather observations.
        date_column: Name of the required date column.
        max_temperature_column: Name of the required maximum-temperature
            column.
        min_temperature_column: Optional minimum-temperature column name.
        humidity_column: Optional humidity column name.
        wind_speed_column: Optional wind-speed column name.
        lag_days: Positive calendar-day offsets for lagged weather predictors.
            Values must be unique positive integers.
        rolling_windows: Positive calendar-day window sizes for maximum-
            temperature rolling mean and rolling maximum. Values must be
            unique positive integers.

    Returns:
        A new DataFrame sorted by date, with normalized dates, generated
        predictor columns, and the source DataFrame's index preserved.

    Raises:
        TypeError: If ``observations`` is not a pandas DataFrame.
        ValueError: If required columns are missing, dates are invalid or
            duplicated by calendar day, a used weather column is non-numeric,
            a configured column name is invalid, or a lag/window is invalid.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if observations.empty:
        raise ValueError("observations must contain at least one daily observation.")

    configured_columns = {
        "date_column": date_column,
        "max_temperature_column": max_temperature_column,
        "min_temperature_column": min_temperature_column,
        "humidity_column": humidity_column,
        "wind_speed_column": wind_speed_column,
    }
    for argument, name in configured_columns.items():
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError(f"{argument} must be a non-empty string or None.")
    if not isinstance(date_column, str) or not date_column:
        raise ValueError("date_column must be a non-empty string.")
    if not isinstance(max_temperature_column, str) or not max_temperature_column:
        raise ValueError("max_temperature_column must be a non-empty string.")

    if min_temperature_column is not None and min_temperature_column == max_temperature_column:
        raise ValueError("Minimum- and maximum-temperature columns must be distinct.")
    optional_columns = (
        min_temperature_column,
        humidity_column,
        wind_speed_column,
    )
    selected_optional_columns = [
        name
        for name in optional_columns
        if name is not None and name in observations.columns
    ]
    selected_weather_columns = [max_temperature_column, *selected_optional_columns]
    if len(set(selected_weather_columns)) != len(selected_weather_columns):
        raise ValueError("Configured weather column names must be distinct.")
    if date_column in selected_weather_columns:
        raise ValueError("date_column must differ from configured weather columns.")

    missing_columns = [
        name
        for name in (date_column, max_temperature_column)
        if name not in observations.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}.")

    def validate_periods(periods: Sequence[int], argument: str) -> list[int]:
        if isinstance(periods, (str, bytes)):
            raise ValueError(f"{argument} must be a sequence of positive integers.")
        try:
            values = list(periods)
        except TypeError as exc:
            raise ValueError(
                f"{argument} must be a sequence of positive integers."
            ) from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ValueError(f"{argument} must contain only positive integers.")
        if len(set(values)) != len(values):
            raise ValueError(f"{argument} must not contain duplicate values.")
        return values

    lags = validate_periods(lag_days, "lag_days")
    windows = validate_periods(rolling_windows, "rolling_windows")
    feature_names = [date_column, "month", "day_of_year"]
    for column in selected_weather_columns:
        feature_names.extend(f"{column}_lag_{lag}" for lag in lags)
    for window in windows:
        feature_names.extend(
            (
                f"{max_temperature_column}_rolling_mean_{window}",
                f"{max_temperature_column}_rolling_max_{window}",
            )
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Configured column names conflict with generated feature names.")

    try:
        parsed_dates = pd.to_datetime(
            observations[date_column],
            errors="coerce",
            format="mixed",
        )
        calendar_dates = parsed_dates.dt.normalize()
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
    duplicate_dates = calendar_dates.duplicated(keep=False)
    if duplicate_dates.any():
        duplicate_values = (
            calendar_dates.loc[duplicate_dates].drop_duplicates().astype(str).tolist()[:5]
        )
        raise ValueError(
            f"Duplicate daily dates in column {date_column!r}: {duplicate_values}."
        )

    source = observations.copy()
    source[date_column] = calendar_dates
    source = source.sort_values(date_column, kind="stable")

    observed_dates = pd.DatetimeIndex(source[date_column])
    calendar = pd.date_range(
        start=observed_dates.min(),
        end=observed_dates.max(),
        freq="D",
    )

    weather: dict[str, pd.Series] = {}
    for column in selected_weather_columns:
        try:
            values = pd.to_numeric(source[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Weather column {column!r} must contain numeric values or missing values."
            ) from exc
        if values.dropna().isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"Weather column {column!r} contains a non-finite value.")
        weather[column] = pd.Series(
            values.to_numpy(),
            index=observed_dates,
        ).reindex(calendar)

    features = pd.DataFrame(index=source.index)
    features[date_column] = observed_dates.to_numpy()

    for column, values in weather.items():
        for lag in lags:
            lagged = values.shift(lag).reindex(observed_dates)
            features[f"{column}_lag_{lag}"] = lagged.to_numpy()

    maximum = weather[max_temperature_column]
    past_maximum = maximum.shift(1)
    for window in windows:
        rolling_mean = past_maximum.rolling(
            window=window,
            min_periods=window,
        ).mean().reindex(observed_dates)
        rolling_maximum = past_maximum.rolling(
            window=window,
            min_periods=window,
        ).max().reindex(observed_dates)
        features[f"{max_temperature_column}_rolling_mean_{window}"] = (
            rolling_mean.to_numpy()
        )
        features[f"{max_temperature_column}_rolling_max_{window}"] = (
            rolling_maximum.to_numpy()
        )

    features["month"] = source[date_column].dt.month
    features["day_of_year"] = source[date_column].dt.dayofyear
    return features