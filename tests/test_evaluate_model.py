import numpy as np
import pandas as pd
import pytest

from src import evaluate_model as ev


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 6, n)               # skewed like a rare event
    y = (rng.random(n) < p).astype(int)   # perfectly calibrated by construction
    return y, p


def test_perfect_predictor_scores_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    r = ev.evaluate_probabilities(y, y.astype(float))
    assert r["pr_auc"] == pytest.approx(1.0) and r["roc_auc"] == pytest.approx(1.0)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0
    assert r["brier"] == pytest.approx(0.0)


def test_confusion_counts_add_up():
    y, p = _synthetic()
    r = ev.evaluate_probabilities(y, p, threshold=0.3)
    assert r["tn"] + r["fp"] + r["fn"] + r["tp"] == len(y)
    assert r["tp"] + r["fn"] == y.sum()


def test_calibrated_probabilities_have_small_ece_and_miscalibrated_large():
    y, p = _synthetic(20000)
    good = ev.expected_calibration_error(ev.calibration_table(y, p, n_bins=10))
    bad = ev.expected_calibration_error(ev.calibration_table(y, np.clip(p * 3, 0, 1), n_bins=10))
    assert good < 0.02 and bad > 3 * good


def test_single_class_gives_nan_aucs_not_an_error():
    r = ev.probability_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert np.isnan(r["roc_auc"]) and np.isnan(r["pr_auc"])


def test_few_positives_flagged_unreliable():
    y = np.zeros(200, int); y[:2] = 1
    r = ev.evaluate_probabilities(y, np.linspace(0, 1, 200))
    assert r["reliable"] is False and "unstable" in r["note"]


def test_nan_pairs_are_dropped_and_bad_inputs_rejected():
    r = ev.probability_metrics([0, 1, np.nan, 1], [0.1, 0.9, 0.5, np.nan])
    assert r["n"] == 2
    with pytest.raises(ValueError):
        ev.probability_metrics([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError):
        ev.probability_metrics([0, 2], [0.1, 0.5])
    with pytest.raises(ValueError):
        ev.probability_metrics([0, 1, 1], [0.1, 0.5])


def test_identical_probabilities_do_not_crash_calibration():
    t = ev.calibration_table([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3])
    assert len(t) == 1 and t.n.iloc[0] == 4


def test_persistence_baseline_is_todays_label():
    assert ev.persistence_baseline([0, 1, 1]).tolist() == [0, 1, 1]


def test_climatology_baseline_uses_training_doy_rate_only():
    idx = pd.date_range("2000-01-01", "2009-12-31")
    y = pd.Series(0.0, index=idx)
    y[(idx.month == 5) & (idx.day == 10) & (idx.year < 2005)] = 1.0   # 5 of 10 years on 10 May
    p = ev.climatology_baseline(y, pd.to_datetime(["2012-05-10", "2011-11-10"]), window=0)
    assert p[0] == pytest.approx(0.5) and p[1] == 0.0
    smooth = ev.climatology_baseline(y, pd.to_datetime(["2012-05-11"]), window=2)
    assert 0 < smooth[0] < 0.5


def test_climatology_requires_datetime_index():
    with pytest.raises(TypeError):
        ev.climatology_baseline(pd.Series([0, 1]), pd.to_datetime(["2020-01-01"]))


def test_compare_models_has_row_per_model_plus_persistence():
    y, p = _synthetic()
    t = ev.compare_models(y, {"a": p, "b": np.full_like(p, y.mean())}, persistence=np.roll(y, 1))
    assert t.model.tolist() == ["a", "b", "persistence"]
    assert t.set_index("model").loc["a", "pr_auc"] > t.set_index("model").loc["b", "pr_auc"]


def test_block_bootstrap_interval_contains_estimate_and_needs_enough_data():
    y, p = _synthetic(1500, seed=3)
    ci = ev.block_bootstrap_ci(y, p, n_boot=150, block=30)
    assert ci["lo95"] <= ci["estimate"] <= ci["hi95"]
    with pytest.raises(ValueError):
        ev.block_bootstrap_ci(y[:20], p[:20], block=30)


def test_plots_return_figures(tmp_path):
    y, p = _synthetic(500)
    t = ev.calibration_table(y, p)
    assert ev.plot_calibration({"m": t}, tmp_path / "c.png") is not None
    assert (tmp_path / "c.png").exists()
    assert ev.plot_pr_curves(y, {"m": p}, tmp_path / "p.png") is not None
