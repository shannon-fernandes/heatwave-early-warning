"""Cost-based warning thresholds and risk levels (research prototype).

A warning is issued when the model's probability of a heatwave-proxy day is at
least ``threshold``. The threshold minimises  cost_missed * (missed events)
+ cost_false_alarm * (false alarms)  on data the model was NOT trained on.
The cost ratio is a stated assumption, not a measured quantity.
"""
from __future__ import annotations

import numpy as np


def choose_threshold(y_true, y_prob, cost_missed: float = 10.0, cost_false_alarm: float = 1.0) -> dict:
    """Return the probability threshold with the lowest total cost.

    Ties go to the higher threshold (fewer warnings). If warning never is cheapest
    (for example no positives), the threshold is just above the largest probability,
    so it can exceed 1.0 and no warning is ever issued.
    """
    if cost_missed <= 0 or cost_false_alarm <= 0:
        raise ValueError("Costs must be positive.")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must have the same length.")
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep].astype(int), p[keep]
    if len(y) == 0:
        raise ValueError("No valid observations.")
    positives = int(y.sum())
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last_of_value = np.r_[ps[:-1] != ps[1:], True]          # warn on all p >= this value
    idx = np.flatnonzero(last_of_value)
    costs = cost_missed * (positives - tp[idx]) + cost_false_alarm * fp[idx]
    never_cost = cost_missed * positives
    best = int(np.argmin(costs))                            # argmin picks the earliest = highest threshold
    if costs[best] < never_cost:
        threshold, cost, warnings = float(ps[idx[best]]), float(costs[best]), int(idx[best] + 1)
    else:
        threshold, cost, warnings = float(ps[0] + 1e-9), float(never_cost), 0
    return {
        "threshold": threshold,
        "cost": cost,
        "n_warnings": warnings,
        "cost_never_warn": float(never_cost),
        "cost_always_warn": float(cost_false_alarm * (len(y) - positives)),
    }


def risk_cutoffs(threshold: float) -> tuple[float, float]:
    """(moderate, high): moderate starts at the cost-based threshold, high halfway from there to 1."""
    high = threshold + (1.0 - threshold) / 2 if threshold < 1.0 else float("inf")
    return float(threshold), float(high)


def risk_level(probability: float, threshold: float) -> str:
    """'Low' below the threshold, 'Moderate' from it, 'High' halfway between it and 1."""
    if probability is None or np.isnan(probability):
        raise ValueError("probability is required.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    moderate, high = risk_cutoffs(threshold)
    if probability >= high:
        return "High"
    if probability >= moderate:
        return "Moderate"
    return "Low"
