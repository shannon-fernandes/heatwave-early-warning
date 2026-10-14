"""Load the project configuration and create output folders."""
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["period"]["end_year"] is None:
        cfg["period"]["end_year"] = date.today().year - 1
    if cfg["period"]["start_year"] > cfg["period"]["end_year"]:
        raise ValueError("start_year must not be after end_year")
    for key in ("raw_dir", "processed_dir", "figures_dir", "tables_dir", "models_dir"):
        (PROJECT_ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Absolute path for a folder listed under `paths` in the config."""
    return PROJECT_ROOT / cfg["paths"][key]


def city_slug(cfg: dict) -> str:
    """'Chennai' -> 'chennai' (used in file names)."""
    return cfg["study_area"]["name"].lower().replace(" ", "_")