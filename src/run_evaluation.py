"""Honest end-to-end evaluation: model vs baselines, calibration, cost-based warnings.

Run:  python -m src.run_evaluation
Reads: data/processed/<city>_dashboard.csv   (from src.export_dashboard_csv)
Writes (outputs/tables, outputs/figures):
  label_counts.csv             events per split for each label definition
  model_comparison_<label>.csv model vs climatology vs persistence on the HOLDOUT half
  calibration_<label>.csv/.png reliability of the model on the holdout half
  pr_curve_<label>.png
Design (small number of events, so be careful):
  * chronological split: train | validation (default 80/20 by date)
  * validation is cut in half by date: the FIRST half picks each model's cost-based
    warning threshold, the SECOND half (holdout) is used only for scoring
  * baselines: climatology (training-years rate for that day of year) and persistence
    (yesterday's label); persistence treats an unknown yesterday as 'no warning'
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import evaluate_model as ev
from src.config import city_slug, get_path, load_config
from src.heatwave_labels import assign_heatwave_proxy_labels
from src.train_model import train_baseline_model
from src.warning_logic import choose_threshold

log = logging.getLogger(__name__)

# Label definitions. The IMD-style rule is kept for transparency: on Chennai it yields only a
# handful of events, so it cannot support a trained model. The relaxed rule is the working target
# and is NOT an official IMD heatwave.
LABEL_SPECS = {
    "severe_hot_day_proxy": dict(max_temperature_threshold_c=38.0, departure_threshold_c=3.5, minimum_consecutive_days=2),
    "imd_style_proxy": dict(max_temperature_threshold_c=37.0, departure_threshold_c=4.5, minimum_consecutive_days=2),
}
# Pre-specified (the dashboard defaults); deliberately NOT tuned on the holdout.
LAGS, WINDOWS, TRAIN_FRACTION = (1, 2, 3, 7), (3, 7), 0.8
COST_MISSED, COST_FALSE_ALARM = 10.0, 1.0


def label_counts(frame: pd.DataFrame, name: str, spec: dict, train_fraction: float = TRAIN_FRACTION) -> pd.DataFrame:
    labels = assign_heatwave_proxy_labels(frame, **spec)["heatwave_proxy"]
    ordered = labels.set_axis(pd.DatetimeIndex(frame["date"])).sort_index()
    split = int(len(ordered) * train_fraction)
    rows = []
    for part, chunk in (("train", ordered.iloc[:split]), ("validation", ordered.iloc[split:])):
        rows.append({"label": name, "split": part, "days": len(chunk), "unknown": int(chunk.isna().sum()),
                     "positive_days": int((chunk == True).sum()),  # noqa: E712 (nullable boolean)
                     "first": chunk.index.min().date(), "last": chunk.index.max().date(), **spec})
    return pd.DataFrame(rows)


def evaluate_label(frame: pd.DataFrame, name: str, spec: dict, n_boot: int = 300) -> dict:
    labels = assign_heatwave_proxy_labels(frame, **spec)["heatwave_proxy"]
    labels = labels.set_axis(pd.DatetimeIndex(frame["date"])).sort_index()
    res = train_baseline_model(frame, **spec, lag_days=LAGS, rolling_windows=WINDOWS, train_fraction=TRAIN_FRACTION,
                               cost_missed=COST_MISSED, cost_false_alarm=COST_FALSE_ALARM,
                               use_departure_features=True)
    y, p, d = res.validation_true, res.validation_probability, res.validation_dates
    train_labels = pd.Series(res.training_true.astype(float), index=res.training_dates)
    clim = ev.climatology_baseline(train_labels, d)
    prev = labels.reindex(pd.date_range(labels.index.min(), labels.index.max())).shift(1)
    prev = prev.reindex(d).astype("Float64").fillna(0).astype(float).to_numpy()   # unknown yesterday -> no warning
    half = max(1, len(y) // 2)
    calib, hold = slice(0, half), slice(half, None)
    models = {"logistic_regression": p, "climatology": clim}
    thresholds = {n: choose_threshold(y[calib], q[calib], COST_MISSED, COST_FALSE_ALARM)["threshold"]
                  for n, q in models.items()}
    table = ev.compare_models(y[hold], {n: q[hold] for n, q in models.items()}, threshold=thresholds,
                              persistence=prev[hold])
    cal = ev.calibration_table(y[hold], p[hold], n_bins=5)
    ci = None
    if len(y[hold]) >= 90 and y[hold].sum() > 0:
        ci = ev.block_bootstrap_ci(y[hold], p[hold], "pr_auc", n_boot=n_boot, block=30)
    return {"result": res, "table": table, "calibration": cal, "pr_auc_ci": ci, "y_hold": y[hold],
            "probs_hold": {n: q[hold] for n, q in models.items()}, "thresholds": thresholds,
            "validation_start": d.min(), "calibration_end": d[half - 1], "holdout_start": d[half], "validation_end": d.max()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    path = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_dashboard.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.export_dashboard_csv` first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    tables, figures = get_path(cfg, "tables_dir"), get_path(cfg, "figures_dir")

    counts = pd.concat([label_counts(frame, n, s) for n, s in LABEL_SPECS.items()])
    counts.to_csv(tables / "label_counts.csv", index=False)
    log.info("Events per split:\n%s", counts[["label", "split", "days", "unknown", "positive_days"]].to_string(index=False))

    for name, spec in LABEL_SPECS.items():
        try:
            out = evaluate_label(frame, name, spec)
        except ValueError as exc:
            log.warning("%s: cannot train/evaluate (%s)", name, exc)
            continue
        out["table"].to_csv(tables / f"model_comparison_{name}.csv", index=False)
        out["calibration"].to_csv(tables / f"calibration_{name}.csv", index=False)
        ev.plot_calibration({name: out["calibration"]}, figures / f"calibration_{name}.png")
        ev.plot_pr_curves(out["y_hold"], out["probs_hold"], figures / f"pr_curve_{name}.png")
        log.info("%s: thresholds chosen on %s to %s, scored on holdout %s to %s",
                 name, out["validation_start"].date(), out["calibration_end"].date(),
                 out["holdout_start"].date(), out["validation_end"].date())
        log.info("\n%s", out["table"].round(3).to_string(index=False))
        if out["pr_auc_ci"]:
            log.info("Model PR-AUC 95%% block-bootstrap CI: %s", {k: round(v, 3) for k, v in out["pr_auc_ci"].items()})
        if not out["table"]["reliable"].all():
            log.warning("%s: fewer than %d positives in the holdout: treat scores as indicative only.", name, ev.MIN_POSITIVES)
    log.info("Wrote tables to %s and figures to %s", tables, figures)


if __name__ == "__main__":
    main()
