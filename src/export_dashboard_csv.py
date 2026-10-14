"""Turn the processed daily table into the CSV the dashboard and ML engine expect.

Run:  python -m src.export_dashboard_csv
Reads:  data/processed/<city>_daily.csv           (from src.data_preprocessing)
Writes: data/processed/<city>_dashboard.csv       (ML years only)
        data/sample/<city>_dashboard_sample.csv   (same file, small enough to commit)

Columns: date, max_temp_c, min_temp_c, departure_from_normal_c, humidity, wind_speed
(humidity/wind only if NASA POWER data was merged in).

departure_from_normal_c = Tmax minus the mean Tmax for that day of year over the
years <= split.train_end_year (a short, fixed reference period, NOT an official
30-year IMD normal). This is documented in the README; years after the reference
period are out-of-sample, so no future information leaks into later rows.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import city_slug, get_path, load_config
from src.pipeline_labels import compute_climatology

log = logging.getLogger(__name__)

RENAME = {"tmax": "max_temp_c", "tmin": "min_temp_c", "rh2m": "humidity", "wind": "wind_speed"}


def build_dashboard_frame(daily: pd.DataFrame, reference_end_year: int, start_year: int, end_year: int) -> pd.DataFrame:
    if "tmax" not in daily.columns:
        raise KeyError(f"'tmax' not in processed file. Columns: {list(daily.columns)}")
    clim = compute_climatology(daily["tmax"], reference_end_year)
    out = daily.copy()
    out["departure_from_normal_c"] = out["tmax"] - clim.reindex(out.index.dayofyear).to_numpy()
    out = out.loc[f"{start_year}-01-01":f"{end_year}-12-31"].rename(columns=RENAME)
    keep = ["max_temp_c", "min_temp_c", "departure_from_normal_c", "humidity", "wind_speed"]
    out = out[[c for c in keep if c in out.columns]]
    out.index.name = "date"
    return out.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    src = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_daily.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found. Run `python -m src.data_preprocessing` first.")
    daily = pd.read_csv(src, index_col="date", parse_dates=True)
    a = cfg["analysis"]
    frame = build_dashboard_frame(daily, cfg["split"]["train_end_year"], a["ml_start_year"], a["ml_end_year"])
    out = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_dashboard.csv"
    frame.to_csv(out, index=False)
    sample_dir = get_path(cfg, "processed_dir").parent / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_dir / f"{city_slug(cfg)}_dashboard_sample.csv"
    frame.to_csv(sample, index=False)
    missing = frame.drop(columns="date").isna().mean().round(3).to_dict()
    log.info("Wrote %d rows (%s to %s) to %s and %s", len(frame), frame.date.min().date(), frame.date.max().date(), out, sample)
    log.info("Missing fraction by column: %s", missing)
    if "humidity" not in frame.columns:
        log.warning("No humidity/wind columns: NASA POWER data was not merged. The model will use temperature features only.")


if __name__ == "__main__":
    main()
