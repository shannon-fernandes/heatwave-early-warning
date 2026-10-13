"""Research proxy labels for daily heatwave observations."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import pandas as pd


def assign_heatwave_proxy_labels(
    observations: pd.DataFrame,
    *,
    date_column: str = "date",
    max_temperature_column: str = "max_temp_c",
    departure_column: str = "departure_from_normal_c",
    label_column: str = "heatwave_proxy",
    max_temperature_threshold_c: float = 37.0,
    departure_threshold_c: float = 4.5,
    minimum_consecutive_days: int = 2,
) -> pd.DataFrame:
    """Return a copy of daily observations with research heatwave proxy labels.

    A day qualifies when both its maximum temperature is at least
    ``max_temperature_threshold_c`` and its temperature departure from normal
    is at least ``departure_threshold_c``. The defaults (37 °C and 4.5 °C)
    are a simplified IMD-style coastal proxy, not an official IMD declaration;
    in particular, this function does not apply spatial declaration
    requirements.

    By default, qualifying conditions must persist for two consecutive
    calendar days. Every qualifying day in a run of at least
    ``minimum_consecutive_days`` days receives ``True``; a shorter run receives
    ``False``. Set the value to 1 to label each individually qualifying day.
    Missing observations, non-consecutive dates, or a missing calendar day
    break a run. Rows with either required temperature value missing receive
    ``pd.NA`` in the nullable-boolean label column: missing departure is never
    substituted or treated as a confirmed positive.

    Args:
        observations: DataFrame containing one daily observation per date.
        date_column: Name of the date column. Distinct dates are interpreted by
            their local calendar day; multiple rows on one day are rejected.
        max_temperature_column: Name of the daily maximum temperature column,
            in degrees Celsius.
        departure_column: Name of the temperature-departure-from-normal column,
            in degrees Celsius.
        label_column: Name of the output nullable-boolean label column. It must
            differ from the three input column names.
        max_temperature_threshold_c: Inclusive maximum-temperature threshold.
        departure_threshold_c: Inclusive departure-from-normal threshold.
        minimum_consecutive_days: Positive number of qualifying consecutive
            calendar days needed for a positive label.

    Returns:
        A copy of ``observations`` with ``label_column`` added. The new column
        has pandas nullable-boolean dtype: ``True`` for a proxy heatwave day,
        ``False`` for a known day not assigned positive, and ``pd.NA`` when
        either required measurement is missing.

    Raises:
        TypeError: If ``observations`` is not a pandas DataFrame.
        ValueError: If required columns are absent, column names conflict,
            dates are invalid or duplicated by calendar day, a measurement is
            non-numeric/non-finite, thresholds are invalid, or the persistence
            requirement is not a positive integer.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if observations.empty:
        raise ValueError("observations must contain at least one daily observation.")

    column_names = {
        "date_column": date_column,
        "max_temperature_column": max_temperature_column,
        "departure_column": departure_column,
        "label_column": label_column,
    }
    for argument, name in column_names.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{argument} must be a non-empty string.")

    input_columns = (date_column, max_temperature_column, departure_column)
    if len(set(input_columns)) != len(input_columns):
        raise ValueError("Date, maximum-temperature, and departure columns must be distinct.")
    if label_column in input_columns:
        raise ValueError("label_column must differ from all input column names.")

    missing_columns = [name for name in input_columns if name not in observations.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}.")

    if (
        isinstance(minimum_consecutive_days, bool)
        or not isinstance(minimum_consecutive_days, int)
        or minimum_consecutive_days < 1
    ):
        raise ValueError("minimum_consecutive_days must be a positive integer.")

    thresholds = {
        "max_temperature_threshold_c": max_temperature_threshold_c,
        "departure_threshold_c": departure_threshold_c,
    }
    for argument, threshold in thresholds.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"{argument} must be a finite number.")
        if not math.isfinite(threshold):
            raise ValueError(f"{argument} must be a finite number.")

    try:
        parsed_dates = pd.to_datetime(observations[date_column], errors="coerce")
        calendar_dates = parsed_dates.dt.normalize()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Column {date_column!r} must contain consistently parseable daily dates."
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

    numeric_columns: dict[str, pd.Series] = {}
    for column in (max_temperature_column, departure_column):
        try:
            numeric = pd.to_numeric(observations[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column {column!r} must contain numeric values or missing values."
            ) from exc

        try:
            finite_values = numeric.dropna().map(math.isfinite)
        except TypeError as exc:
            raise ValueError(
                f"Column {column!r} must contain real numeric values or missing values."
            ) from exc
        if not finite_values.all():
            raise ValueError(f"Column {column!r} contains a non-finite measurement.")
        numeric_columns[column] = numeric

    maximum = numeric_columns[max_temperature_column]
    departure = numeric_columns[departure_column]
    known_measurements = maximum.notna() & departure.notna()
    qualifies = (
        maximum.ge(max_temperature_threshold_c)
        & departure.ge(departure_threshold_c)
        & known_measurements
    )

    count = len(observations)
    labels = pd.array([pd.NA] * count, dtype="boolean")
    known_positions = known_measurements.to_numpy(dtype=bool)
    labels[known_positions] = False
    qualifies_by_position = qualifies.to_numpy(dtype=bool)
    date_by_position = calendar_dates.to_numpy()
    ordered_positions = sorted(
        range(count),
        key=lambda position: date_by_position[position],
    )

    qualifying_run: list[int] = []
    previous_date: Optional[date] = None

    def label_completed_run() -> None:
        if len(qualifying_run) >= minimum_consecutive_days:
            for position in qualifying_run:
                labels[position] = True

    for position in ordered_positions:
        current_date = pd.Timestamp(date_by_position[position]).date()
        if not known_positions[position] or not qualifies_by_position[position]:
            label_completed_run()
            qualifying_run = []
            previous_date = None
            continue

        if (
            previous_date is not None
            and current_date - previous_date != timedelta(days=1)
        ):
            label_completed_run()
            qualifying_run = []

        qualifying_run.append(position)
        previous_date = current_date

    label_completed_run()

    result = observations.copy()
    result[label_column] = pd.array(labels, dtype="boolean")
    return result