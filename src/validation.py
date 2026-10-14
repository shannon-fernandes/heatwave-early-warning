"""Shared input validation for the dashboard-side modules (labels, features)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_daily_dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    """Return normalised calendar dates aligned to ``frame``'s index.

    Raises ValueError for a missing column, invalid/missing dates, or duplicate calendar days."""
    if date_column not in frame.columns:
        raise ValueError(f"Missing required column: {date_column!r}.")
    try:
        parsed = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {date_column!r} must contain consistently parseable dates.") from exc
    invalid = parsed.isna()
    if invalid.any():
        rows = frame.index[invalid].tolist()[:5]
        raise ValueError(
            f"Column {date_column!r} contains invalid or missing dates at row index(es): {rows}."
        )
    normalized = parsed.dt.normalize()
    if normalized.duplicated(keep=False).any():
        raise ValueError(f"Duplicate daily dates in column {date_column!r}.")
    return normalized


def require_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return ``frame[column]`` as float with blanks as NaN; raise ValueError if absent or non-numeric."""
    if column not in frame.columns:
        raise ValueError(f"Missing required column: {column!r}.")
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} must contain numeric values or blanks.") from exc
    return values.astype("float64").replace([np.inf, -np.inf], np.nan)
