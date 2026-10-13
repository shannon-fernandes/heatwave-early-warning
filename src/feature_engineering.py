"""Leakage-safe features and chronological splits.

Run (after Person C's labels exist):  python -m src.feature_engineering
Input:  data/processed/<city>_daily_labeled.csv  (needs a 0/1 column `heatwave`)
Output: data/processed/<city>_features_h<horizon>.csv
"""
import logging

import numpy as np
import pandas as pd

from src.config import city_slug, get_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def climatology_by_doy(tmax: pd.Series, train_end_year: int, smooth_days: int = 15) -> pd.Series:
    """Smoothed mean Tmax per day of year (1..366), using TRAINING YEARS ONLY."""
    train = tmax[tmax.index.year <= train_end_year]
    clim = train.groupby(train.index.dayofyear).mean().reindex(range(1, 367))
    clim = clim.interpolate(limit_direction="both")
    ext = pd.concat([clim.iloc[-smooth_days:], clim, clim.iloc[:smooth_days]])  # wrap around the year
    smooth = ext.rolling(smooth_days, center=True, min_periods=1).mean()
    return smooth.iloc[smooth_days:-smooth_days]


def build_features(df: pd.DataFrame, label_col: str, horizon: int, train_end_year: int,
                   base_columns, lag_columns, lag_days, rolling_windows) -> pd.DataFrame:
    """Return a table of features (known at the end of day t) plus `target`
    (= label on day t + horizon). Rows with any missing value are dropped."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 for a forecast")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted by date")
    steps = df.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(days=1)).all():
        raise ValueError("index must be a complete daily calendar (run preprocessing)")
    needed = set(base_columns) | set(lag_columns) | {"tmax", "tmin", label_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    feat = pd.DataFrame(index=df.index)
    for c in base_columns:
        feat[c] = df[c]
    feat["diurnal_range"] = df["tmax"] - df["tmin"]
    for c in lag_columns:
        for k in lag_days:
            feat[f"{c}_lag{k}"] = df[c].shift(k)                 # shift(k>0) = past only
    for w in rolling_windows:                                    # window ends at day t
        feat[f"tmax_mean{w}"] = df["tmax"].rolling(w, min_periods=w).mean()
        feat[f"tmax_max{w}"] = df["tmax"].rolling(w, min_periods=w).max()

    clim = climatology_by_doy(df["tmax"], train_end_year)
    anom = df["tmax"] - clim.reindex(df.index.dayofyear).to_numpy()
    feat["tmax_anom"] = anom
    feat["tmax_anom_mean3"] = anom.rolling(3, min_periods=3).mean()

    doy = df.index.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)           # calendar is known in advance
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feat["label_today"] = df[label_col]                          # today's status, known at issue time
    feat["target"] = df[label_col].shift(-horizon)               # the ONLY use of future data

    feat = feat.dropna()
    feat["target"] = feat["target"].astype(int)
    return feat


def chronological_split(feat: pd.DataFrame, train_end_year: int, val_end_year: int, horizon: int):
    """Train / validation / test by date, with a purge of `horizon` days at the end of
    train and validation so no target date falls inside the next period."""
    if not train_end_year < val_end_year:
        raise ValueError("train_end_year must be before val_end_year")
    train_end = pd.Timestamp(f"{train_end_year}-12-31")
    val_end = pd.Timestamp(f"{val_end_year}-12-31")
    gap = pd.Timedelta(days=horizon)
    train = feat[feat.index <= train_end - gap]
    val = feat[(feat.index > train_end) & (feat.index <= val_end - gap)]
    test = feat[feat.index > val_end]
    return train, val, test


def build_features_from_config(df: pd.DataFrame, cfg: dict, label_col: str = "heatwave") -> pd.DataFrame:
    f = cfg["features"]
    return build_features(df, label_col, cfg["forecast"]["horizon_days"],
                          cfg["split"]["train_end_year"], f["base_columns"],
                          f["lag_columns"], f["lag_days"], f["rolling_windows"])


def main() -> None:
    cfg = load_config()
    path = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_daily_labeled.csv"
    if not path.exists():
        log.error("%s not found. Please run heatwave_labels.py to create it first.", path)
        raise SystemExit(1)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    a = cfg["analysis"]
    df = df.loc[f"{a['ml_start_year']}-01-01":f"{a['ml_end_year']}-12-31"]   # POWER era, grid years only
    h = cfg["forecast"]["horizon_days"]
    feat = build_features_from_config(df, cfg)
    out = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_features_h{h}.csv"
    feat.to_csv(out)
    log.info("Saved %d rows, %d features to %s", len(feat), feat.shape[1] - 1, out)
    s = cfg["split"]
    for name, part in zip(("train", "val", "test"),
                          chronological_split(feat, s["train_end_year"], s["val_end_year"], h)):
        log.info("%-5s rows=%d  heatwave targets=%d  (%.2f%%)", name, len(part),
                 int(part["target"].sum()), 100 * part["target"].mean() if len(part) else 0)

if __name__ == "__main__":
    main()