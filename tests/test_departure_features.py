import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_features
from src.predict_warning import predict_warning
from src.train_model import train_baseline_model


def _frame(n=60):
    hot = {2, 3, 8, 9, 14, 15, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 55, 56}
    return pd.DataFrame({
        "date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "max_temp_c": [38.0 if d in hot else 32.0 + d % 4 for d in range(n)],
        "departure_from_normal_c": [5.0 if d in hot else 1.0 + d % 3 for d in range(n)],
    })


def test_departure_lags_are_optional_and_past_only():
    f = _frame()
    off = build_features(f, lag_days=(1, 2), rolling_windows=())
    on = build_features(f, lag_days=(1, 2), rolling_windows=(), departure_column="departure_from_normal_c")
    assert not any("departure" in c for c in off.columns)
    assert on.loc[5, "departure_from_normal_c_lag_1"] == f.loc[4, "departure_from_normal_c"]
    changed = f.copy()
    changed.loc[10:, "departure_from_normal_c"] = 99.0           # future values must not touch earlier rows
    on2 = build_features(changed, lag_days=(1, 2), rolling_windows=(), departure_column="departure_from_normal_c")
    pd.testing.assert_frame_equal(on.loc[:10], on2.loc[:10])


def test_missing_departure_column_is_a_clear_error():
    with pytest.raises(ValueError, match="Missing required column"):
        build_features(_frame().drop(columns="departure_from_normal_c"), departure_column="departure_from_normal_c")


def test_training_and_prediction_use_departure_features_without_target_day_leak():
    f = _frame()
    res = train_baseline_model(f, train_fraction=0.7, lag_days=(1, 2), rolling_windows=(2,),
                               minimum_validation_positive_cases=1, use_departure_features=True)
    assert res.departure_column == "departure_from_normal_c"
    assert any("departure" in c for c in res.feature_columns)
    assert res.decision_threshold is not None and len(res.validation_true) == len(res.validation_probability)
    target = pd.Timestamp("2024-06-01")
    hist = f[f.date < target]
    row = pd.DataFrame({"date": [target], "max_temp_c": [np.nan], "departure_from_normal_c": [np.nan]})
    base = pd.concat([hist, row], ignore_index=True)
    leaky = base.copy()
    leaky.loc[leaky.index[-1], ["max_temp_c", "departure_from_normal_c"]] = [99.0, 99.0]
    a = predict_warning(res, base, target, lag_days=(1, 2), rolling_windows=(2,))
    b = predict_warning(res, leaky, target, lag_days=(1, 2), rolling_windows=(2,))
    assert a.positive_class_probability == b.positive_class_probability
    assert 0 <= a.positive_class_probability <= 1
    assert all(isinstance(n, str) and v > 0 for n, v in a.contributing_factors)


def test_prediction_without_departure_column_fails_clearly_when_model_needs_it():
    f = _frame()
    res = train_baseline_model(f, train_fraction=0.7, lag_days=(1,), rolling_windows=(), use_departure_features=True,
                               minimum_validation_positive_cases=1)
    target = pd.Timestamp("2024-06-01")
    obs = pd.concat([f[f.date < target][["date", "max_temp_c"]], pd.DataFrame({"date": [target], "max_temp_c": [np.nan]})],
                    ignore_index=True)
    with pytest.raises(ValueError, match="Missing required column"):
        predict_warning(res, obs, target, lag_days=(1,), rolling_windows=())
