"""Build a dashboard CSV that runs up to the present, from ONE source (NASA POWER).

Run:  python -m src.update_live
Writes: data/raw/nasa_power_live.csv            (raw download)
        data/processed/<city>_dashboard_live.csv (1981 to the latest published day)

Why POWER only: the historical file mixes OpenCity (IMD grid) Tmax with POWER humidity/wind.
IMD data stops in 2024 and switches to a station record in 2021, so it cannot be extended to
today. Using POWER for every variable keeps training and prediction consistent. The price:
POWER is a coarse reanalysis, its Tmax runs lower than IMD's, so event counts differ from the
OpenCity-based analysis. Re-run `python -m src.run_evaluation --input <live csv>` and report
that result separately.

Latency: POWER publishes with a delay of some days. The newest rows with no temperature yet are
dropped, and the dashboard predicts the day AFTER the last available day, which may be
several days before today. This is a statistical next-day model on observed data, not a
numerical weather forecast.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import city_slug, get_path, load_config
from src.pipeline_labels import compute_climatology

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RENAME = {"T2M_MAX": "max_temp_c", "T2M_MIN": "min_temp_c", "RH2M": "humidity", "WS2M": "wind_speed"}
COLUMNS = ["date", "max_temp_c", "min_temp_c", "departure_from_normal_c", "humidity", "wind_speed"]
BACK_OFF_DAYS = (1, 4, 10, 30)          # end dates to try, in days before today


def parse_power_payload(payload: dict) -> pd.DataFrame:
    """POWER JSON -> tidy frame. -999 is POWER's fill value; trailing days without a
    temperature are not yet published and are dropped."""
    try:
        data = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected NASA POWER response: {str(payload)[:300]}") from exc
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index().replace(-999, np.nan).rename(columns=RENAME)
    for need in ("max_temp_c", "min_temp_c"):
        if need not in df.columns:
            raise RuntimeError(f"{need} missing from the POWER response; columns: {list(df.columns)}")
    last = df["max_temp_c"].last_valid_index()
    if last is None:
        raise RuntimeError("NASA POWER returned no temperature values for the requested period.")
    df = df.loc[:last]
    df.index.name = "date"
    return df[[c for c in RENAME.values() if c in df.columns]].reset_index()


def fetch_power(start: date, lat: float, lon: float, community: str = "AG", today: date | None = None,
                timeout: int = 180) -> pd.DataFrame:
    """Download daily data from `start` to the newest available day (tries a few end dates)."""
    today = today or date.today()
    last_error = "no attempt made"
    for back in BACK_OFF_DAYS:
        end = today - timedelta(days=back)
        params = {"parameters": ",".join(RENAME), "community": community, "longitude": lon, "latitude": lat,
                  "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"}
        log.info("Requesting NASA POWER %s to %s ...", start, end)
        try:
            r = requests.get(POWER_URL, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach NASA POWER: {exc}") from exc
        if r.ok:
            return parse_power_payload(r.json())
        last_error = f"HTTP {r.status_code}: {r.text[:200]}"
        log.warning("End date %s refused (%s); trying an earlier one.", end, last_error)
    raise RuntimeError(f"NASA POWER refused every end date tried. Last error: {last_error}")


def with_departure(power: pd.DataFrame, reference_end_year: int) -> pd.DataFrame:
    """Add departure_from_normal_c: Tmax minus the day-of-year mean Tmax over years <= reference_end_year
    (a short fixed reference period, not an official 30-year normal)."""
    s = power.set_index(pd.DatetimeIndex(power["date"]))["max_temp_c"]
    clim = compute_climatology(s, reference_end_year)
    out = power.copy()
    out["departure_from_normal_c"] = s.to_numpy() - clim.reindex(s.index.dayofyear).to_numpy()
    return out[[c for c in COLUMNS if c in out.columns]]


def recent_gaps(frame: pd.DataFrame, days: int = 8) -> int:
    """Missing values among the last `days` rows (the model looks back up to 7 days)."""
    tail = frame.tail(days).drop(columns="date")
    return int(tail.isna().sum().sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    s = cfg["study_area"]
    raw = fetch_power(date(cfg["analysis"]["ml_start_year"], 1, 1), s["latitude"], s["longitude"],
                      cfg.get("nasa_power", {}).get("community", "AG"))
    raw.to_csv(get_path(cfg, "raw_dir") / "nasa_power_live.csv", index=False)
    frame = with_departure(raw, cfg["split"]["train_end_year"])
    out = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_dashboard_live.csv"
    frame.to_csv(out, index=False)
    latest = pd.Timestamp(frame["date"].max())
    behind = (pd.Timestamp(date.today()) - latest).days
    log.info("Wrote %d rows (%s to %s) to %s", len(frame), frame["date"].min().date(), latest.date(), out)
    log.info("Latest observation is %d day(s) old. The dashboard can predict %s.", behind,
             (latest + pd.Timedelta(days=1)).date())
    gaps = recent_gaps(frame)
    if gaps:
        log.warning("%d missing value(s) in the last 8 days: those predictions lean on imputed values.", gaps)


if __name__ == "__main__":
    main()
