import numpy as np
import pytest

from src.warning_logic import choose_threshold, risk_cutoffs, risk_level


def test_threshold_minimises_cost_on_a_clean_example():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    r = choose_threshold(y, p, cost_missed=10, cost_false_alarm=1)
    assert r["cost"] == 0 and 0.4 < r["threshold"] <= 0.8 and r["n_warnings"] == 2


def test_costly_misses_lower_the_threshold():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(int)
    cheap = choose_threshold(y, p, cost_missed=1)["threshold"]
    dear = choose_threshold(y, p, cost_missed=50)["threshold"]
    assert dear < cheap


def test_no_positives_means_never_warn():
    r = choose_threshold([0, 0, 0], [0.2, 0.5, 0.9])
    assert r["n_warnings"] == 0 and r["threshold"] > 0.9 and r["cost"] == 0


def test_ties_go_to_the_higher_threshold():
    r = choose_threshold([1, 0], [0.5, 0.5], cost_missed=1, cost_false_alarm=1)
    assert r["n_warnings"] == 0          # warning on both costs 1, never warning costs 1: prefer fewer warnings


def test_bad_inputs():
    with pytest.raises(ValueError):
        choose_threshold([1], [0.5, 0.6])
    with pytest.raises(ValueError):
        choose_threshold([np.nan], [np.nan])
    with pytest.raises(ValueError):
        choose_threshold([1], [0.5], cost_missed=0)


def test_risk_levels_and_cutoffs():
    assert risk_cutoffs(0.5) == (0.5, 0.75)
    assert [risk_level(p, 0.5) for p in (0.2, 0.5, 0.74, 0.75, 1.0)] == ["Low", "Moderate", "Moderate", "High", "High"]
    assert risk_level(0.99, 1.2) == "Low"          # sentinel threshold above 1: never warn
    with pytest.raises(ValueError):
        risk_level(1.5, 0.5)
