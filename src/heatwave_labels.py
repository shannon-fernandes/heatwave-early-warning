"""Compute heatwave labels based on IMD criteria.

Run: python -m src.heatwave_labels
Input: data/processed/<city>_daily.csv
Output: data/processed/<city>_daily_labeled.csv
"""
import logging
import numpy as np
import pandas as pd

from src.config import city_slug, get_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_climatology(tmax: pd.Series, baseline_end_year: int) -> pd.Series:
    """Compute mean Tmax per day-of-year using baseline training years."""
    train = tmax[tmax.index.year <= baseline_end_year]
    clim = train.groupby(train.index.dayofyear).mean().reindex(range(1, 367))
    return clim.interpolate(limit_direction="both")


def label_heatwaves(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Label days as heatwave (1) or not (0) using IMD criteria.
    
    For coastal stations like Chennai:
    - Absolute threshold: Tmax >= 37°C
    - Departure threshold: Tmax departure from normal >= 4.5°C
    """
    out = df.copy()
    train_end = cfg["split"]["train_end_year"]
    
    # Compute baseline climatology
    clim = compute_climatology(out["tmax"], train_end)
    out["normal_tmax"] = clim.reindex(out.index.dayofyear).to_numpy()
    out["tmax_departure"] = out["tmax"] - out["normal_tmax"]
    
    # IMD Coastal criteria
    abs_threshold = 37.0 if cfg["study_area"]["region_type"] == "coastal" else 40.0
    departure_threshold = 4.5
    
    hw_condition = (out["tmax"] >= abs_threshold) & (out["tmax_departure"] >= departure_threshold)
    out["heatwave"] = hw_condition.astype(int)
    
    log.info("Labeled %d heatwave days out of %d total days (%.2f%%)", 
             out["heatwave"].sum(), len(out), 100 * out["heatwave"].mean())
    return out


def main() -> None:
    cfg = load_config()
    processed_dir = get_path(cfg, "processed_dir")
    input_path = processed_dir / f"{city_slug(cfg)}_daily.csv"
    
    if not input_path.exists():
        log.error("%s not found. Run `python -m src.data_preprocessing` first.", input_path)
        raise SystemExit(1)
        
    df = pd.read_csv(input_path, index_col="date", parse_dates=True)
    df_labeled = label_heatwaves(df, cfg)
    
    output_path = processed_dir / f"{city_slug(cfg)}_daily_labeled.csv"
    df_labeled.to_csv(output_path)
    log.info("Saved labeled daily data to %s", output_path)


if __name__ == "__main__":
    main()