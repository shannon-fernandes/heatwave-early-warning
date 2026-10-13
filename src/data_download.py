"""Download the daily data used by the project.

Sources
- OpenCity Chennai daily Tmax/Tmin 1951-2024 (primary temperature record)
- NASA POWER daily humidity, dew point and wind (1981 onwards)

Run:  python -m src.data_download [--source opencity|nasa|all] [--force]
"""
import argparse
import logging

import numpy as np
import pandas as pd
import requests

from src.config import get_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
EXPECTED_OPENCITY_COLUMNS = ["Date", "Rain", "Temp Max", "Temp Min"]


def download_opencity(cfg: dict, force: bool = False):
    oc = cfg["opencity"]
    out = get_path(cfg, "raw_dir") / oc["filename"]
    if out.exists() and not force:
        log.info("%s already exists, skipping (use --force to re-download)", out.name)
        return out
    log.info("Downloading %s", oc["url"])
    try:
        r = requests.get(oc["url"], timeout=120)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Download failed: {exc}\nManual fallback: open the dataset page, click Download, "
            f"and save the CSV as {out}") from exc
    text = r.content.decode("utf-8-sig")             # strips a BOM if present
    header = [c.strip() for c in text.splitlines()[0].split(",")]
    if header != EXPECTED_OPENCITY_COLUMNS:
        raise RuntimeError(f"Unexpected columns {header}; expected {EXPECTED_OPENCITY_COLUMNS}. "
                           "The dataset may have changed: inspect it before continuing.")
    out.write_bytes(text.encode("utf-8"))
    log.info("Saved %d data rows to %s", len(text.splitlines()) - 1, out)
    log.info("Attribution: %s", oc["credit"])
    return out


def download_nasa_power(cfg: dict, force: bool = False):
    out = get_path(cfg, "raw_dir") / "nasa_power_daily.csv"
    if out.exists() and not force:
        log.info("%s already exists, skipping (use --force to re-download)", out.name)
        return out
    start = cfg["nasa_power"]["start_year"]
    end = cfg["period"]["end_year"]
    if start < 1981:
        raise ValueError("NASA POWER daily data starts in 1981; set nasa_power.start_year >= 1981")
    params = {
        "parameters": ",".join(cfg["nasa_power"]["parameters"]),
        "community": cfg["nasa_power"]["community"],
        "longitude": cfg["study_area"]["longitude"],
        "latitude": cfg["study_area"]["latitude"],
        "start": f"{start}0101",
        "end": f"{end}1231",
        "format": "JSON",
    }
    log.info("Requesting NASA POWER %s-%s ...", start, end)
    r = requests.get(POWER_URL, params=params, timeout=180)
    if not r.ok:
        raise RuntimeError(f"NASA POWER returned {r.status_code}: {r.text[:500]}")
    payload = r.json()
    try:
        data = payload["properties"]["parameter"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected NASA POWER response: {str(payload)[:500]}") from exc
    df = pd.DataFrame(data)                          # rows = YYYYMMDD strings
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index().replace(-999, np.nan)       # -999 is POWER's documented fill value
    df.to_csv(out)
    log.info("Saved %d rows to %s", len(df), out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["opencity", "nasa", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    config = load_config()
    if args.source in ("opencity", "all"):
        download_opencity(config, args.force)
    if args.source in ("nasa", "all"):
        download_nasa_power(config, args.force)