"""Tests for leakage-safe weather predictor construction."""

import pandas as pd
import pytest

from src.feature_engineering import build_features


def _daily_weather() -> pd.DataFrame:
    """Return synthetic daily weather with identifiable source-row indexes."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-05-01", periods=5, freq="D"),
            "max_temp_c": [30.0, 32.0, 31.0, 35.0, 36.0],
            "min_temp_c": [20.0, 21.0, 22.0, 23.0, 24.0],
            "humidity": [50.0, 51.0, 52.0, 53.0, 54.0],
            "wind_speed": [2.0, 3.0, 4.0, 5.0, 6.0],
            "heatwave_target": [0, 0, 0, 1, 1],
        },
        index=[10, 20, 30, 40, 50],
    )


def test_lags_use_previous_observations_and_preserve_date_mapping() -> None:
    observations = _daily_weather()

    features = build_features(
        observations,
        lag_days=(1, 2),
        rolling_windows=(),
    )

    assert features.index.tolist() == [10, 20, 30, 40, 50]
    assert features["date"].tolist() == observations["date"].tolist()
    assert pd.isna(features.loc[10, "max_temp_c_lag_1"])
    assert features.loc[20, "max_temp_c_lag_1"] == 30.0
    assert features.loc[30, "max_temp_c_lag_1"] == 32.0
    assert features.loc[30, "max_temp_c_lag_2"] == 30.0
    assert features.loc[30, "min_temp_c_lag_1"] == 21.0
    assert features.loc[30, "humidity_lag_1"] == 51.0
    assert features.loc[30, "wind_speed_lag_1"] == 3.0
    assert "heatwave_target" not in features.columns


def test_rolling_features_use_prior_observations_only() -> None:
    observations = _daily_weather()

    features = build_features(
        observations,
        lag_days=(),
        rolling_windows=(2, 3),
    )

    assert pd.isna(features.loc[10, "max_temp_c_rolling_mean_2"])
    assert pd.isna(features.loc[20, "max_temp_c_rolling_max_2"])
    # May 3 uses May 1 and May 2, not May 3's own 31 °C.
    assert features.loc[30, "max_temp_c_rolling_mean_2"] == 31.0
    assert features.loc[30, "max_temp_c_rolling_max_2"] == 32.0
    assert features.loc[40, "max_temp_c_rolling_mean_3"] == pytest.approx(31.0)
    assert features.loc[40, "max_temp_c_rolling_max_3"] == 32.0


def test_future_temperature_changes_do_not_change_earlier_features() -> None:
    observations = _daily_weather()
    changed_future = observations.copy()
    changed_future.loc[40:, "max_temp_c"] = [100.0, 200.0]

    original_features = build_features(
        observations,
        lag_days=(1, 2),
        rolling_windows=(2, 3),
    )
    changed_features = build_features(
        changed_future,
        lag_days=(1, 2),
        rolling_windows=(2, 3),
    )

    earlier_indexes = [10, 20, 30]
    pd.testing.assert_frame_equal(
        original_features.loc[earlier_indexes],
        changed_features.loc[earlier_indexes],
    )


def test_unsorted_input_is_sorted_for_calculation_and_keeps_source_indexes() -> None:
    observations = _daily_weather().iloc[[3, 0, 4, 1, 2]]

    features = build_features(
        observations,
        lag_days=(1,),
        rolling_windows=(),
    )

    assert features.index.tolist() == [10, 20, 30, 40, 50]
    assert features["date"].tolist() == list(pd.date_range("2024-05-01", periods=5))
    assert features.loc[30, "max_temp_c_lag_1"] == 32.0


def test_missing_calendar_days_produce_missing_calendar_lags_and_windows() -> None:
    observations = pd.DataFrame(
        {
            "date": [
                "2024-05-01",
                "2024-05-03",
                "2024-05-04",
                "2024-05-05",
            ],
            "max_temp_c": [30.0, 34.0, 35.0, 36.0],
        }
    )

    features = build_features(
        observations,
        lag_days=(1,),
        rolling_windows=(2,),
    )

    assert pd.isna(features.loc[1, "max_temp_c_lag_1"])
    assert pd.isna(features.loc[1, "max_temp_c_rolling_mean_2"])
    assert features.loc[2, "max_temp_c_lag_1"] == 34.0
    assert pd.isna(features.loc[2, "max_temp_c_rolling_max_2"])
    assert features.loc[3, "max_temp_c_rolling_mean_2"] == pytest.approx(34.5)
    assert features.loc[3, "max_temp_c_rolling_max_2"] == 35.0


def test_future_prediction_row_needs_no_target_day_weather() -> None:
    observations = _daily_weather()
    history = observations.iloc[:4].drop(columns="heatwave_target")
    target_row = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-05-05")],
            "max_temp_c": [None],
            "min_temp_c": [None],
            "humidity": [None],
            "wind_speed": [None],
        }
    )
    prediction_input = pd.concat([history, target_row], ignore_index=True)

    historical_features = build_features(
        observations.drop(columns="heatwave_target"),
        lag_days=(1, 2),
        rolling_windows=(2,),
    )
    prediction_features = build_features(
        prediction_input,
        lag_days=(1, 2),
        rolling_windows=(2,),
    )

    assert prediction_features.columns.tolist() == historical_features.columns.tolist()
    assert prediction_features.iloc[-1]["max_temp_c_lag_1"] == 35.0
    assert prediction_features.iloc[-1]["max_temp_c_rolling_mean_2"] == 33.0


def test_empty_observations_raise_clear_error() -> None:
    observations = pd.DataFrame(columns=["date", "max_temp_c"])

    with pytest.raises(ValueError, match="at least one daily observation"):
        build_features(observations)


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (
            [
                pd.Timestamp("2024-05-01"),
                pd.Timestamp("2024-05-01 12:00"),
            ],
            "Duplicate daily dates",
        ),
        (["2024-05-01", "invalid-date"], "invalid or missing dates"),
    ],
    ids=["duplicate-calendar-date", "invalid-date"],
)
def test_duplicate_and_invalid_dates_raise_clear_errors(
    dates: list[object],
    message: str,
) -> None:
    observations = pd.DataFrame(
        {"date": dates, "max_temp_c": [30.0, 31.0]}
    )

    with pytest.raises(ValueError, match=message):
        build_features(observations)


def test_missing_temperatures_remain_missing_in_lags_and_rolling_features() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.date_range("2024-05-01", periods=5, freq="D"),
            "max_temp_c": [30.0, None, 34.0, 35.0, 36.0],
        }
    )

    features = build_features(
        observations,
        lag_days=(1,),
        rolling_windows=(2,),
    )

    assert features.loc[1, "max_temp_c_lag_1"] == 30.0
    assert pd.isna(features.loc[2, "max_temp_c_lag_1"])
    assert pd.isna(features.loc[2, "max_temp_c_rolling_mean_2"])
    assert pd.isna(features.loc[2, "max_temp_c_rolling_max_2"])
    assert pd.isna(features.loc[3, "max_temp_c_rolling_mean_2"])
    assert pd.isna(features.loc[3, "max_temp_c_rolling_max_2"])
    assert features.loc[4, "max_temp_c_rolling_mean_2"] == pytest.approx(34.5)
    assert features.loc[4, "max_temp_c_rolling_max_2"] == 35.0


def test_absent_optional_humidity_and_wind_columns_are_skipped() -> None:
    observations = pd.DataFrame(
        {
            "date": pd.date_range("2024-05-01", periods=2, freq="D"),
            "max_temp_c": [30.0, 31.0],
        }
    )

    features = build_features(
        observations,
        lag_days=(1,),
        rolling_windows=(),
    )

    assert "max_temp_c_lag_1" in features.columns
    assert not any(name.startswith("humidity_") for name in features.columns)
    assert not any(name.startswith("wind_speed_") for name in features.columns)
    assert not any(name.startswith("min_temp_c_") for name in features.columns)


def test_missing_required_columns_raise_clear_error() -> None:
    observations = pd.DataFrame({"date": ["2024-05-01"], "humidity": [50.0]})

    with pytest.raises(ValueError, match="Missing required column"):
        build_features(observations)


def test_build_features_does_not_modify_input_dataframe() -> None:
    observations = _daily_weather().iloc[[2, 0, 1]]
    original = observations.copy(deep=True)

    result = build_features(
        observations,
        lag_days=(1,),
        rolling_windows=(2,),
    )

    pd.testing.assert_frame_equal(observations, original)
    assert "max_temp_c_lag_1" not in observations.columns
    assert "max_temp_c_lag_1" in result.columns
