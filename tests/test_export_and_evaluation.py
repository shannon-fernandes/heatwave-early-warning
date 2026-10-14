import numpy as np
import pandas as pd

from src import run_evaluation as R
from src.export_dashboard_csv import build_dashboard_frame


def _daily(years=12, seed=0, start=1990):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start}-01-01", periods=int(years * 365.25), freq="D", name="date")
    doy = idx.dayofyear.to_numpy()
    t = 33 + 4 * np.sin((doy - 80) / 365.25 * 2 * np.pi) + rng.gumbel(0, 1.2, len(idx))
    return pd.DataFrame({"tmax": t, "tmin": t - 8, "rain": np.nan, "source": "imd_grid", "rh2m": 70.0, "wind": 3.0}, index=idx)


def test_export_columns_years_and_reference_period():
    d = _daily()
    out = build_dashboard_frame(d, reference_end_year=1995, start_year=1992, end_year=2000)
    assert list(out.columns) == ["date", "max_temp_c", "min_temp_c", "departure_from_normal_c", "humidity", "wind_speed"]
    assert out.date.min() == pd.Timestamp("1992-01-01") and out.date.max() == pd.Timestamp("2000-12-31")
    ref = d.loc[:"1995-12-31", "tmax"]
    clim = ref.groupby(ref.index.dayofyear).mean()
    row = out.iloc[100]
    expected = row.max_temp_c - clim.reindex([pd.Timestamp(row.date).dayofyear]).iloc[0]
    assert abs(row.departure_from_normal_c - expected) < 1e-9


def test_export_without_power_columns_still_works():
    d = _daily().drop(columns=["rh2m", "wind"])
    out = build_dashboard_frame(d, 1995, 1992, 2000)
    assert "humidity" not in out.columns and "departure_from_normal_c" in out.columns


def test_evaluation_runs_end_to_end_on_synthetic_data():
    d = _daily(years=30, seed=2)
    frame = build_dashboard_frame(d, 1999, 1990, 2019)
    out = R.evaluate_label(frame, "severe_hot_day_proxy", dict(max_temperature_threshold_c=38.0, departure_threshold_c=3.0,
                           minimum_consecutive_days=2), n_boot=30)
    t = out["table"].set_index("model")
    assert set(t.index) == {"logistic_regression", "climatology", "persistence"}
    assert (t["n"] == t["n"].iloc[0]).all()
    assert out["holdout_start"] > out["calibration_end"]          # thresholds never see the holdout
    counts = R.label_counts(frame, "x", dict(max_temperature_threshold_c=38.0, departure_threshold_c=3.0, minimum_consecutive_days=2))
    assert counts.days.sum() == len(frame)
