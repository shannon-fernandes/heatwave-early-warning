"""Evaluation for the heatwave classifier (Person B, Parts 2 and 3).

Pure functions on arrays/Series, so they work with any model that outputs
probabilities. Accuracy is deliberately not reported: heatwave days are rare.

Typical use (y_true / y_prob are the validation or test targets and the
model's probability of class 1, aligned by date):

    res = evaluate_probabilities(y_true, y_prob, threshold=0.5)
    table = compare_models(y_true, {"logreg": p_lr, "climatology": p_clim},
                           persistence=label_today)

CLI (a predictions CSV with columns date,y_true,y_prob[,label_today]):

    python -m src.evaluate_model --predictions path/to/predictions.csv
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)
MIN_POSITIVES = 5  # below this, scores are flagged as unreliable


def _clean(y_true, y_prob=None):
    y = np.asarray(y_true, dtype=float)
    if y_prob is None:
        keep = ~np.isnan(y)
        return y[keep].astype(int), None
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must have the same length.")
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep].astype(int), p[keep]
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true must be binary (0/1).")
    if len(p) and (p.min() < 0 or p.max() > 1):
        raise ValueError("y_prob must be probabilities in [0, 1].")
    return y, p


# ------------------------------------------------------------------ metrics
def threshold_metrics(y_true, y_pred) -> dict:
    y, p = _clean(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, p.astype(int), labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def probability_metrics(y_true, y_prob) -> dict:
    """PR-AUC (average precision), ROC-AUC, Brier score. AUCs are NaN if only one class is present."""
    y, p = _clean(y_true, y_prob)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }


def calibration_table(y_true, y_prob, n_bins: int = 5, strategy: str = "quantile") -> pd.DataFrame:
    """Reliability table: mean predicted vs observed frequency per bin.
    Quantile bins keep a usable count per bin when positives are rare."""
    y, p = _clean(y_true, y_prob)
    if len(y) == 0:
        raise ValueError("No valid observations.")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")
    if len(edges) < 2:  # all probabilities identical
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    b = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.any():
            rows.append({"bin": k, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_frequency": float(y[m].mean()), "n_positive": int(y[m].sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    n = table["n"].sum()
    return float((table["n"] / n * (table["mean_predicted"] - table["observed_frequency"]).abs()).sum())


def evaluate_probabilities(y_true, y_prob, threshold: float = 0.5, n_bins: int = 5) -> dict:
    """Everything for one model at one decision threshold."""
    y, p = _clean(y_true, y_prob)
    out = {"threshold": threshold}
    out.update(probability_metrics(y, p))
    out.update(threshold_metrics(y, (p >= threshold).astype(int)))
    cal = calibration_table(y, p, n_bins=n_bins)
    out["ece"] = expected_calibration_error(cal)
    out["calibration"] = cal
    out["reliable"] = out["n_positive"] >= MIN_POSITIVES
    out["note"] = None if out["reliable"] else (
        f"Only {out['n_positive']} positive case(s): scores are unstable, treat as indicative only.")
    return out


# ---------------------------------------------------------------- baselines
def persistence_baseline(label_today) -> np.ndarray:
    """Predict that the heatwave status tomorrow equals today's. Hard 0/1 prediction."""
    return np.asarray(label_today, dtype=float)


def _doy365(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365-day calendar: 10 May is the same key in leap and non-leap years
    (29 Feb shares the key of 28 Feb)."""
    doy = index.dayofyear.to_numpy().copy()
    leap_after_feb = index.is_leap_year & (doy > 59)
    doy[leap_after_feb] -= 1
    return doy


def climatology_baseline(train_labels: pd.Series, target_dates, window: int = 7) -> np.ndarray:
    """Probability = share of TRAINING years with a heatwave on that day of year
    (circular +/- `window` day smoothing). `target_dates` are the dates the
    labels refer to (issue date + horizon). Uses only training labels, so it is
    leakage-free if train_labels covers the training period only."""
    if window < 0:
        raise ValueError("window must be >= 0.")
    s = train_labels.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("train_labels needs a DatetimeIndex.")
    rate = s.groupby(_doy365(s.index)).mean().reindex(range(1, 366))
    if window > 0:
        ext = pd.concat([rate.iloc[-window:], rate, rate.iloc[:window]])
        rate = ext.rolling(2 * window + 1, center=True, min_periods=1).mean().iloc[window:-window]
    rate = rate.fillna(float(s.mean()))
    return rate.reindex(_doy365(pd.DatetimeIndex(target_dates))).to_numpy()


# ------------------------------------------------------------ comparison table
def compare_models(y_true, probs: dict, threshold=0.5, persistence=None) -> pd.DataFrame:
    """One row per model. `threshold` is one number or a {model: threshold} dict. `persistence` (today's label) adds a persistence row
    (hard predictions, so AUCs are computed on the 0/1 values and are not comparable with probabilities)."""
    rows = []
    items = dict(probs)
    if persistence is not None:
        items["persistence"] = persistence_baseline(persistence)
    for name, p in items.items():
        th = threshold.get(name, 0.5) if isinstance(threshold, dict) else threshold
        r = evaluate_probabilities(y_true, p, threshold=th)
        r["threshold"] = th
        rows.append({"model": name, **{k: r[k] for k in (
            "n", "n_positive", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier", "ece",
            "tn", "fp", "fn", "tp", "reliable")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------- uncertainty (few events)
def block_bootstrap_ci(y_true, y_prob, metric: str = "pr_auc", n_boot: int = 500,
                       block: int = 30, seed: int = 0) -> dict:
    """Moving-block bootstrap CI for a probability metric. Blocks keep
    day-to-day dependence; with very few positives expect a very wide interval."""
    y, p = _clean(y_true, y_prob)
    n = len(y)
    if n < 2 * block:
        raise ValueError("Series too short for the chosen block length.")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        m = probability_metrics(y[idx], p[idx])[metric]
        if not np.isnan(m):
            vals.append(m)
    point = probability_metrics(y, p)[metric]
    if not vals:
        return {"estimate": point, "lo95": float("nan"), "hi95": float("nan"), "n_valid": 0}
    return {"estimate": point, "lo95": float(np.percentile(vals, 2.5)),
            "hi95": float(np.percentile(vals, 97.5)), "n_valid": len(vals)}


# ----------------------------------------------------------------------- plots
def plot_calibration(tables: dict, path=None):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, t in tables.items():
        ax.plot(t["mean_predicted"], t["observed_frequency"], "o-", label=name)
    ax.set(xlabel="mean predicted probability", ylabel="observed frequency", title="Calibration (reliability)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_pr_curves(y_true, probs: dict, path=None):
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve

    y, _ = _clean(y_true)
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    for name, p in probs.items():
        yy, pp = _clean(y_true, p)
        pr, rc, _ = precision_recall_curve(yy, pp)
        ax.plot(rc, pr, label=name)
    ax.axhline(y.mean(), color="k", ls="--", label=f"base rate {y.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# ------------------------------------------------------------------------- CLI
def main() -> None:
    from src.config import get_path, load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="CSV with date,y_true,y_prob[,label_today]")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions, parse_dates=["date"]).set_index("date").sort_index()
    for c in ("y_true", "y_prob"):
        if c not in df.columns:
            raise SystemExit(f"{args.predictions} must contain column {c!r}")
    cfg = load_config()
    tables, figures = get_path(cfg, "tables_dir"), get_path(cfg, "figures_dir")
    persistence = df["label_today"] if "label_today" in df.columns else None
    table = compare_models(df["y_true"], {"model": df["y_prob"]}, args.threshold, persistence)
    table.to_csv(tables / "model_comparison.csv", index=False)
    res = evaluate_probabilities(df["y_true"], df["y_prob"], args.threshold)
    res["calibration"].to_csv(tables / "calibration_table.csv", index=False)
    plot_calibration({"model": res["calibration"]}, figures / "calibration.png")
    plot_pr_curves(df["y_true"], {"model": df["y_prob"]}, figures / "pr_curve.png")
    log.info("\n%s", table.round(3).to_string(index=False))
    if res["note"]:
        log.warning(res["note"])
    log.info("Wrote model_comparison.csv, calibration_table.csv, calibration.png, pr_curve.png")


if __name__ == "__main__":
    main()
