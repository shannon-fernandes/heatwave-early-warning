from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import dashboard as D

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def test_pretty_feature_names():
    assert D.pretty_feature("departure_from_normal_c_lag_1") == "Departure from normal, yesterday"
    assert D.pretty_feature("humidity_lag_3") == "Humidity, 3 days ago"
    assert D.pretty_feature("max_temp_c_rolling_mean_7") == "Average max temperature, previous 7 days"
    assert D.pretty_feature("max_temp_c_rolling_max_3") == "Highest max temperature, previous 3 days"
    assert D.pretty_feature("month") == "Month of year"
    assert D.pretty_feature("something_else") == "something_else"


def test_parse_rejects_bad_files():
    with pytest.raises(ValueError, match="missing required"):
        D.parse_weather_csv(b"date,max_temp_c\n2020-01-01,30\n")
    with pytest.raises(ValueError, match="duplicate"):
        D.parse_weather_csv(b"date,max_temp_c,departure_from_normal_c\n2020-01-01,30,1\n2020-01-01,31,1\n")
    with pytest.raises(ValueError, match="numeric"):
        D.parse_weather_csv(b"date,max_temp_c,departure_from_normal_c\n2020-01-01,hot,1\n")
    ok = D.parse_weather_csv(b"date,max_temp_c,departure_from_normal_c\n2020-01-02,31,1\n2020-01-01,,1\n")
    assert list(ok[D.DATE]) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")] and ok[D.TMAX].isna().sum() == 1


def test_candidate_dates_end_one_day_after_the_data():
    idx = pd.date_range("2020-01-01", "2020-03-31")
    ds = pd.DataFrame({D.DATE: idx, D.TMAX: 30.0, D.DEP: 0.0})

    class R:                                        # minimal stand-in for TrainingResult
        training_end = pd.Timestamp("2020-02-15")

    dates = D.candidate_dates(ds, R(), {"lag_days": (1, 7), "rolling_windows": (3,)})
    assert dates[0] == pd.Timestamp("2020-02-16") and dates[-1] == pd.Timestamp("2020-04-01")


def test_full_flow_on_builtin_sample_without_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Overview", "How extreme?", "Model check", "Predict a day", "Guide"]
    assert any("days loaded" in c.value for c in at.caption)
    at.button(key="train_btn").click().run()
    assert not at.exception, at.exception
    assert any("real severe hot days" in s.value for s in at.success)
    at.button(key="predict_btn").click().run()
    assert not at.exception, at.exception
    assert any(" risk" in m.value for m in at.markdown)          # the coloured badge
    assert any("Chance of a severe hot day" in m.label for m in at.metric)


def test_missing_live_file_gives_a_helpful_message(tmp_path, monkeypatch):
    """Independent of what is on disk: point the live option at a file that does not exist."""
    live = "Live data (up to the latest published day)"
    monkeypatch.setitem(D.DATA_CHOICES, live, tmp_path / "chennai_dashboard_live.csv")
    dataset, message = D.load_dataset(live, None)
    assert dataset is None and "update_live" in message


def test_live_option_loads_when_the_file_exists(tmp_path, monkeypatch):
    live = "Live data (up to the latest published day)"
    f = tmp_path / "chennai_dashboard_live.csv"
    f.write_text("date,max_temp_c,departure_from_normal_c\n2026-01-01,31.0,0.5\n2026-01-02,32.0,1.0\n")
    monkeypatch.setitem(D.DATA_CHOICES, live, f)
    dataset, message = D.load_dataset(live, None)
    assert message == "" and len(dataset) == 2


def test_upload_option_waits_for_a_file():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio(key="data_choice").set_value("Upload my own CSV").run()
    assert not at.exception, at.exception
    assert any("Upload a daily CSV" in i.value for i in at.info)
