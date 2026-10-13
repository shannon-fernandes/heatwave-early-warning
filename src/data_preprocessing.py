"""Clean, validate and merge raw data into one daily table.

Run:  python -m src.data_preprocessing
Output: data/processed/<city>_daily.csv
        outputs/tables/data_quality_report.csv
        outputs/tables/source_break_report.csv   (is there a jump at the grid->station switch?)
        outputs/tables/source_comparison.csv     (OpenCity vs NASA POWER Tmax)

Design choices
- No interpolation: it would use future values (leakage) and invent extremes.
- Out-of-range values become NaN and are counted, never silently kept.
- A `source` column records which series each day comes from, because the
  OpenCity file switches from IMD grid data to a station record in 2021.
"""
import logging

import numpy as np
import pandas as pd

from src.config import city_slug, get_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POWER_RENAME = {"T2M_MAX": "power_tmax", "T2M_MIN": "power_tmin",
                "RH2M": "rh2m", "T2MDEW": "dewpoint", "WS2M": "wind"}


def find_column(df: pd.DataFrame, candidates, what: str) -> str:
    """Return the first matching column name (case-insensitive) or fail clearly."""
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise KeyError(f"Could not find {what}. Tried {list(candidates)}. "
                   f"Columns present: {list(df.columns)}")


def load_opencity(path, date_format: str = "%d-%m-%Y") -> pd.DataFrame:
    """Read the OpenCity file into numeric columns tmax, tmin, rain with a date index."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = find_column(df, ["Date"], "date column")
    tmax_col = find_column(df, ["Temp Max", "tmax"], "daily maximum temperature")
    tmin_col = find_column(df, ["Temp Min", "tmin"], "daily minimum temperature")
    try:
        rain_col = find_column(df, ["Rain", "rain"], "rainfall")
    except KeyError:
        rain_col = None

    dates = pd.to_datetime(df[date_col].astype(str).str.strip(), format=date_format, errors="coerce")
    if dates.isna().any():
        bad = df.loc[dates.isna(), date_col].iloc[0]
        raise ValueError(f"{int(dates.isna().sum())} dates do not match {date_format}; first: {bad!r}")

    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for name, col in (("tmax", tmax_col), ("tmin", tmin_col), ("rain", rain_col)):
        if col is None:
            out[name] = np.nan
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & num.isna()
        if bad.any():
            log.warning("%s: %d non-numeric entries treated as missing", name, int(bad.sum()))
        out[name] = num.to_numpy(dtype="float64")
    return out.sort_index()


def load_nasa_power(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = find_column(df, ["date"], "NASA POWER date column")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index().rename_axis("date")
    df = df.rename(columns=POWER_RENAME)
    if "power_tmax" not in df.columns or "power_tmin" not in df.columns:
        raise KeyError(f"T2M_MAX / T2M_MIN missing. Columns present: {list(df.columns)}")
    return df[[c for c in POWER_RENAME.values() if c in df.columns]]


def label_source(index: pd.DatetimeIndex, station_from_year: int) -> pd.Series:
    """'imd_grid' before the switch year, 'nungambakkam_station' from it onwards."""
    values = np.where(index.year >= station_from_year, "nungambakkam_station", "imd_grid")
    return pd.Series(values, index=index, name="source")


def make_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete daily calendar so missing days show up as NaN rows."""
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found; inspect the raw files.")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full).rename_axis("date")


def apply_range_checks(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in ranges.items():
        if col not in out.columns:
            continue
        bad = out[col].notna() & ~out[col].between(lo, hi)
        if bad.any():
            log.warning("%s: %d values outside [%s, %s] set to NaN", col, int(bad.sum()), lo, hi)
            out.loc[bad, col] = np.nan
    return out


def fix_inverted_extremes(df: pd.DataFrame, hi: str, lo: str) -> pd.DataFrame:
    """Where daily max < daily min the row is physically impossible: set both to NaN."""
    out = df.copy()
    bad = out[hi] < out[lo]
    if bad.any():
        log.warning("%s < %s on %d days; both set to NaN", hi, lo, int(bad.sum()))
        out.loc[bad, [hi, lo]] = np.nan
    return out


def enforce_missing_limit(df: pd.DataFrame, cols, max_fraction: float) -> None:
    for c in cols:
        frac = df[c].isna().mean()
        log.info("%s missing fraction: %.3f", c, frac)
        if frac > max_fraction:
            raise ValueError(f"{c} is {frac:.1%} missing (limit {max_fraction:.1%}). "
                             "Inspect the data before continuing.")


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({"column": c, "n_days": len(s), "n_missing": int(s.isna().sum()),
                     "pct_missing": round(100 * s.isna().mean(), 2),
                     "min": s.min(), "max": s.max(),
                     "first_valid": s.first_valid_index(), "last_valid": s.last_valid_index()})
    return pd.DataFrame(rows)


def source_break_report(df: pd.DataFrame, station_from_year: int, years_before: int = 6) -> pd.DataFrame:
    """Year-by-year summary around the grid->station switch.
    pct_one_decimal: share of Tmax values with at most one decimal place. Gridded values carry
    long float noise (e.g. 29.01000023), so a jump in this share would support a real source change."""
    rows = []
    for y in range(station_from_year - years_before, int(df.index.year.max()) + 1):
        s = df.loc[df.index.year == y, "tmax"].dropna()
        if s.empty:
            continue
        one_dec = np.isclose(s * 10, np.round(s * 10), atol=1e-6).mean()
        rows.append({"year": y, "source": "station" if y >= station_from_year else "grid",
                     "n_days": len(s), "mean_tmax": round(s.mean(), 3),
                     "annual_max": round(s.max(), 2), "pct_one_decimal": round(100 * one_dec, 1)})
    return pd.DataFrame(rows)


def compare_sources(df: pd.DataFrame) -> pd.DataFrame:
    """How different are NASA POWER and OpenCity daily Tmax? Overall and per source."""
    pair = df[["tmax", "power_tmax", "source"]].dropna()
    rows = []
    for name, g in [("all", pair)] + list(pair.groupby("source")):
        diff = g["power_tmax"] - g["tmax"]
        rows.append({"subset": name, "n_days": len(g),
                     "mean_bias_power_minus_opencity": diff.mean(),
                     "rmse": float(np.sqrt((diff ** 2).mean())),
                     "correlation": g["power_tmax"].corr(g["tmax"])})
    return pd.DataFrame(rows)


def build_processed(cfg: dict) -> pd.DataFrame:
    raw = get_path(cfg, "raw_dir")
    pp, oc = cfg["preprocessing"], cfg["opencity"]

    temp_path = raw / oc["filename"]
    if not temp_path.exists():
        raise FileNotFoundError(f"{temp_path} not found. Run `python -m src.data_download` first.")
    df = load_opencity(temp_path, oc["date_format"])

    power_path = raw / "nasa_power_daily.csv"
    if power_path.exists():
        df = df.join(load_nasa_power(power_path), how="left")
    else:
        log.warning("%s not found: humidity/wind columns missing (needed for ML features).", power_path.name)

    p = cfg["period"]
    df = df.loc[f"{p['start_year']}-01-01":f"{p['end_year']}-12-31"]
    df = make_daily_index(df)
    df = apply_range_checks(df, pp["plausible_ranges"])
    df = fix_inverted_extremes(df, "tmax", "tmin")
    if {"power_tmax", "power_tmin"} <= set(df.columns):
        df = fix_inverted_extremes(df, "power_tmax", "power_tmin")
    enforce_missing_limit(df, ["tmax", "tmin"], pp["max_missing_fraction"])

    df.insert(3, "source", label_source(df.index, oc["station_from_year"]))
    first = ["tmax", "tmin", "rain", "source"]
    df = df[first + [c for c in df.columns if c not in first]]

    for name, g in df.groupby("source"):
        log.info("%-22s %d days, tmax missing %.2f%%", name, len(g), 100 * g["tmax"].isna().mean())

    tables = get_path(cfg, "tables_dir")
    quality_report(df.drop(columns="source")).to_csv(tables / "data_quality_report.csv", index=False)
    brk = source_break_report(df, oc["station_from_year"])
    brk.to_csv(tables / "source_break_report.csv", index=False)
    log.info("Source-break check:\n%s", brk.to_string(index=False))
    if "power_tmax" in df.columns:
        comp = compare_sources(df)
        comp.to_csv(tables / "source_comparison.csv", index=False)
        log.info("OpenCity vs POWER Tmax:\n%s", comp.to_string(index=False))

    out = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_daily.csv"
    df.to_csv(out)
    log.info("Saved %d days x %d columns to %s", len(df), df.shape[1], out)
    return df


if __name__ == "__main__":
    build_processed(load_config())