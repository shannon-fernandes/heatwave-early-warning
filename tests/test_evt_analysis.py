import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import evt_analysis as evt


def _daily(n_years=60, seed=0, start=1960):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start}-01-01", periods=int(n_years * 365.25), freq="D")
    doy = idx.dayofyear.to_numpy()
    base = 33 + 4 * np.sin((doy - 80) / 365.25 * 2 * np.pi)
    return pd.Series(base + rng.gumbel(0, 1.0, len(idx)), index=idx, name="tmax")


def test_gev_recovers_known_params():
    x = stats.genextreme.rvs(c=-0.1, loc=40, scale=1.2, size=2000, random_state=1)
    f = evt.fit_gev(x)
    assert f.xi == pytest.approx(0.1, abs=0.05)
    assert f.loc == pytest.approx(40, abs=0.2)
    assert f.scale == pytest.approx(1.2, abs=0.15)


def test_gev_sign_convention_matches_scipy():
    # return level from our wrapper must equal scipy's ppf with c = -xi
    rl = evt.gev_return_level(0.2, 40, 1, 50)
    assert rl == pytest.approx(stats.genextreme.ppf(0.98, c=-0.2, loc=40, scale=1))


def test_return_levels_increase_with_period():
    f = evt.fit_gev(stats.genextreme.rvs(-0.05, 40, 1, size=70, random_state=2))
    rl = f.return_level([2, 10, 50, 100])
    assert np.all(np.diff(rl) > 0)


def test_return_period_must_exceed_one():
    with pytest.raises(ValueError):
        evt.gev_return_level(0.0, 40, 1, 1)


def test_fit_gev_needs_enough_data():
    with pytest.raises(ValueError):
        evt.fit_gev([40, 41, 42])


def test_block_maxima_drops_sparse_years_and_non_summer():
    s = _daily(10)
    s.loc["1962-10-01"] = 99.0  # outside Mar-Jun: must not become a maximum
    s = s[~((s.index.year == 1965) & (s.index.month >= 4))]  # leave 1965 with ~31 summer days
    bm = evt.block_maxima(s)
    assert 1965 not in bm.index
    assert bm.max() < 99
    assert len(bm) == 9


def test_decluster_keeps_one_peak_per_spell():
    idx = pd.date_range("2000-01-01", periods=30)
    v = np.full(30, 30.0)
    v[3:7] = [41, 43, 42, 41]  # spell 1
    v[20:22] = [42, 44]  # spell 2
    peaks = evt.decluster_runs(pd.Series(v, index=idx), 40, run_length=3)
    assert sorted(peaks.tolist()) == [43, 44]


def test_decluster_splits_spells_separated_by_long_gap():
    idx = pd.date_range("2000-01-01", periods=20)
    v = np.full(20, 30.0)
    v[2], v[10] = 41, 42  # 7 non-exceeding days between
    assert len(evt.decluster_runs(pd.Series(v, index=idx), 40, run_length=3)) == 2


def test_decluster_empty_when_no_exceedance():
    s = pd.Series([30.0, 31.0], index=pd.date_range("2000-01-01", periods=2))
    assert evt.decluster_runs(s, 40).empty


def test_gpd_fit_and_return_level_finite_and_monotone():
    s = _daily(60)
    f = evt.fit_gpd(summer_s := evt.summer_only(s), threshold=np.percentile(summer_s, 97), n_years=60)
    assert f.n_peaks >= 15
    rl = f.return_level([5, 10, 50])
    assert np.all(np.isfinite(rl)) and np.all(np.diff(rl) > 0)
    assert rl[0] > f.threshold


def test_gpd_gumbel_limit_continuous():
    f1 = evt.GPDFit(1e-9, 1.0, 40, 50, 1.0)
    f2 = evt.GPDFit(1e-4, 1.0, 40, 50, 1.0)
    assert evt.gpd_return_level(f1, 20) == pytest.approx(evt.gpd_return_level(f2, 20), abs=1e-2)


def test_gpd_too_high_threshold_raises():
    s = _daily(20)
    with pytest.raises(ValueError):
        evt.fit_gpd(s, threshold=s.max() - 0.01)


def test_gev_and_gpd_return_levels_roughly_agree():
    s = _daily(80, seed=3)
    bm = evt.block_maxima(s)
    gev50 = evt.fit_gev(bm).return_level(50)
    summer = evt.summer_only(s)
    gpd = evt.fit_gpd(s, np.percentile(summer.dropna(), 97), n_years=80)
    # rate is per calendar year, summer-only exceedances, so this is comparable
    assert abs(float(gev50) - float(gpd.return_level(50))) < 1.5


def test_bootstrap_ci_contains_estimate_and_is_ordered():
    bm = stats.genextreme.rvs(-0.05, 40, 1, size=60, random_state=4)
    out = evt.bootstrap_gev_return_levels(bm, n_boot=100)
    assert (out.lo95 <= out.estimate).all() and (out.estimate <= out.hi95).all()
    assert (out.hi95 - out.lo95).is_monotonic_increasing  # wider for longer periods


def test_trend_test_detects_trend_and_not_noise():
    yrs = np.arange(1951, 2021)
    rng = np.random.default_rng(11)
    trend = pd.Series(40 + 0.03 * (yrs - 1951) + rng.normal(0, 0.3, len(yrs)), index=yrs)
    noise = pd.Series(40 + rng.normal(0, 0.3, len(yrs)), index=yrs)
    assert evt.trend_test(trend)["p_value"] < 0.01
    assert evt.trend_test(trend)["slope_per_decade"] == pytest.approx(0.3, abs=0.1)
    assert evt.trend_test(noise)["p_value"] > 0.05


def test_exceedance_days_per_year():
    idx = pd.date_range("2000-01-01", "2001-12-31")
    s = pd.Series(30.0, index=idx)
    s.loc["2000-05-01":"2000-05-03"] = 41
    out = evt.empirical_exceedance_by_year(s, 40)
    assert out[2000] == 3 and out[2001] == 0


def test_loader_handles_dashes(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Date,Rain,Temp Max,Temp Min\n01-01-1951,,29.0,20\n02-01-1951,,-----,20\n")
    s = evt.load_raw_tmax(str(p))
    assert s.iloc[0] == 29.0 and np.isnan(s.iloc[1])


# ---------------------------------------------------------- additions (trend GEV, pipeline)
def test_trend_gev_detects_location_trend_and_not_stationary_data():
    yrs = np.arange(1951, 2021)
    rng = np.random.default_rng(21)
    trending = pd.Series(stats.genextreme.rvs(-0.1, 41 + 0.03 * (yrs - 1951), 1.0, random_state=rng), index=yrs)
    flat = pd.Series(stats.genextreme.rvs(-0.1, 41, 1.0, size=len(yrs), random_state=rng), index=yrs)
    t = evt.fit_gev_trend(trending)
    assert t.p_value < 0.01 and t.slope_per_decade == pytest.approx(0.3, abs=0.2)
    assert evt.fit_gev_trend(flat).p_value > 0.01


def test_trend_return_level_rises_with_year_and_period():
    f = evt.TrendGEVFit(xi=-0.1, loc0=41, loc_slope=0.03, scale=1.0, ref_year=1951, lr_stat=10, p_value=0.001)
    assert evt.trend_return_level(f, 2020, 10) > evt.trend_return_level(f, 1951, 10)
    assert evt.trend_return_level(f, 2000, 50) > evt.trend_return_level(f, 2000, 10)
    with pytest.raises(ValueError):
        evt.trend_return_level(f, 2000, 1)


def test_trend_gev_needs_enough_years():
    with pytest.raises(ValueError):
        evt.fit_gev_trend(pd.Series(np.arange(10.0) + 40, index=np.arange(1990, 2000)))


def test_restrict_years_and_processed_loader(tmp_path):
    idx = pd.date_range("2018-01-01", "2022-12-31", name="date")
    df = pd.DataFrame({"tmax": 30.0, "tmin": 20.0, "source": "x"}, index=idx)
    df.loc["2019-05-05", "tmax"] = np.nan
    p = tmp_path / "chennai_daily.csv"
    df.to_csv(p)
    s = evt.load_processed_tmax(str(p))
    assert s.name == "tmax" and s.isna().sum() == 1
    r = evt.restrict_years(s, start=2019, end=2020)
    assert r.index.year.min() == 2019 and r.index.year.max() == 2020
    with pytest.raises(KeyError):
        evt.load_processed_tmax(str(p), column="nope")


def test_run_analysis_writes_outputs(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    cfg = {
        "study_area": {"name": "Testville"},
        "analysis": {"homogeneous_end_year": 2030},
        "paths": {"processed_dir": "p", "tables_dir": "t", "figures_dir": "f"},
    }
    for d in "ptf":
        (tmp_path / d).mkdir()
    s = _daily(70, seed=8, start=1951)
    s.index.name = "date"
    s.to_frame("tmax").to_csv(tmp_path / "p" / "testville_daily.csv")
    out = evt.run_analysis(cfg, n_boot=30)
    assert set(out) == {"primary", "full"}
    assert (tmp_path / "t" / "gev_parameters.csv").exists()
    assert (tmp_path / "t" / "gev_return_levels.csv").exists()
    assert (tmp_path / "f" / "gev_return_levels.png").exists()
    ci = pd.read_csv(tmp_path / "t" / "gev_return_levels.csv")
    assert {2, 5, 10, 20, 50, 100} <= set(ci["T"])


def test_run_analysis_missing_processed_file_message(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / "p").mkdir()
    cfg = {"study_area": {"name": "X"}, "analysis": {"homogeneous_end_year": 2020},
           "paths": {"processed_dir": "p", "tables_dir": "t", "figures_dir": "f"}}
    with pytest.raises(FileNotFoundError, match="data_preprocessing"):
        evt.run_analysis(cfg)
