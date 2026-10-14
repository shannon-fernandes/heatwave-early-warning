"""Heatwave proxy labels for the dashboard/ML engine.

A simplified IMD-style coastal proxy, NOT an official IMD declaration: a day
qualifies when Tmax >= a threshold AND departure from normal >= a threshold.
A day is labelled positive when it sits in a run of at least
``minimum_consecutive_days`` consecutive qualifying calendar days.

Days with a missing Tmax or departure are UNKNOWN (``pd.NA``), never negative,
and they break persistence runs, as do missing calendar days.

(The config-driven batch pipeline version lives in ``pipeline_labels.py``.)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.validation import parse_daily_dates, require_numeric


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
    """Return a copy of ``observations`` with a nullable boolean label column.

    Row order and the original date column are preserved; the input is not modified.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame.")
    if observations.empty:
        raise ValueError("observations must contain at least one daily observation.")
    for name, value in (("max_temperature_threshold_c", max_temperature_threshold_c),
                        ("departure_threshold_c", departure_threshold_c)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if (isinstance(minimum_consecutive_days, bool) or not isinstance(minimum_consecutive_days, int)
            or minimum_consecutive_days < 1):
        raise ValueError("minimum_consecutive_days must be a positive integer.")

    dates = parse_daily_dates(observations, date_column)
    tmax = require_numeric(observations, max_temperature_column)
    departure = require_numeric(observations, departure_column)

    order = np.argsort(dates.to_numpy(), kind="stable")
    d = dates.to_numpy()[order]
    t = tmax.to_numpy()[order]
    dep = departure.to_numpy()[order]

    known = ~(np.isnan(t) | np.isnan(dep))
    qualifying = known & (t >= max_temperature_threshold_c) & (dep >= departure_threshold_c)

    positive = np.zeros(len(d), dtype=bool)
    start = None
    for i in range(len(d) + 1):
        continues = (
            i < len(d) and qualifying[i]
            and (start is None or d[i] - d[i - 1] == np.timedelta64(1, "D"))
        )
        if continues:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= minimum_consecutive_days:
                positive[start:i] = True
            start = None
        if i < len(d) and qualifying[i]:  # a new run begins after a calendar gap
            start = i

    labels_sorted = np.where(known, positive, None)
    labels = np.empty(len(d), dtype=object)
    labels[order] = labels_sorted
    out = observations.copy()
    out[label_column] = pd.array(labels, dtype="boolean")
    return out
