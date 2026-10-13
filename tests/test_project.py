"""Tests for project-level data and labeling behavior."""

import pandas as pd
import pytest

from src.heatwave_labels import assign_heatwave_proxy_labels


def _observations(
    dates: list[object],
    max_temperatures: list[object],
    departures: list[object],
) -> pd.DataFrame:
    """Build a small synthetic observation table."""
    return pd.DataFrame(
        {
            "date": dates,
            "max_temp_c": max_temperatures,
            "departure_from_normal_c": departures,
        }
    )


def test_inclusive_default_thresholds_label_two_qualifying_days() -> None:
    observations = _observations(
        ["2024-05-01", "2024-05-02"],
        [37.0, 38.0],
        [4.5, 5.0],
    )

    result = assign_heatwave_proxy_labels(observations)

    assert result["heatwave_proxy"].tolist() == [True, True]


def test_days_meeting_only_one_default_threshold_are_not_labeled() -> None:
    observations = _observations(
        ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04"],
        [38.0, 36.9, 38.0, 36.9],
        [4.4, 5.0, 4.4, 5.0],
    )

    result = assign_heatwave_proxy_labels(observations)

    assert result["heatwave_proxy"].tolist() == [False, False, False, False]


@pytest.mark.parametrize(
    ("max_temperatures", "departures", "missing_position"),
    [
        ([38.0, None, 38.0], [5.0, 5.0, 5.0], 1),
        ([38.0, 38.0, 38.0], [5.0, None, 5.0], 1),
    ],
)
def test_missing_measurement_is_unknown_and_breaks_persistence(
    max_temperatures: list[object],
    departures: list[object],
    missing_position: int,
) -> None:
    observations = _observations(
        ["2024-05-01", "2024-05-02", "2024-05-03"],
        max_temperatures,
        departures,
    )

    labels = assign_heatwave_proxy_labels(observations)["heatwave_proxy"]

    assert pd.isna(labels.iloc[missing_position])
    assert not labels.iloc[0]
    assert not labels.iloc[2]


def test_shorter_run_than_configured_persistence_is_false() -> None:
    observations = _observations(
        ["2024-05-01", "2024-05-02"],
        [38.0, 38.0],
        [5.0, 5.0],
    )

    labels = assign_heatwave_proxy_labels(
        observations,
        minimum_consecutive_days=3,
    )["heatwave_proxy"]

    assert labels.tolist() == [False, False]


def test_unordered_input_is_labeled_by_consecutive_calendar_day() -> None:
    observations = _observations(
        ["2024-05-03", "2024-05-01", "2024-05-02"],
        [38.0, 37.0, 37.0],
        [5.0, 4.5, 4.5],
    )

    result = assign_heatwave_proxy_labels(observations)

    assert result["heatwave_proxy"].tolist() == [True, True, True]
    assert result["date"].tolist() == observations["date"].tolist()


@pytest.mark.parametrize(
    "dates",
    [
        [
            pd.Timestamp("2024-05-01"),
            pd.Timestamp("2024-05-01 12:00"),
        ],
        ["2024-05-01", "not-a-date"],
    ],
    ids=["duplicate-calendar-date", "invalid-date"],
)
def test_duplicate_or_invalid_dates_raise_clear_validation_error(
    dates: list[object],
) -> None:
    observations = _observations(dates, [38.0, 38.0], [5.0, 5.0])

    expected_message = (
        "Duplicate daily dates"
        if "12:00" in str(dates[1])
        else "invalid or missing dates"
    )
    with pytest.raises(ValueError, match=expected_message):
        assign_heatwave_proxy_labels(observations)


def test_missing_required_column_raises_clear_validation_error() -> None:
    observations = _observations(["2024-05-01"], [38.0], [5.0]).drop(
        columns="departure_from_normal_c"
    )

    with pytest.raises(ValueError, match="Missing required column"):
        assign_heatwave_proxy_labels(observations)


@pytest.mark.parametrize("column", ["max_temp_c", "departure_from_normal_c"])
def test_non_numeric_temperature_value_raises_clear_validation_error(
    column: str,
) -> None:
    observations = _observations(["2024-05-01"], [38.0], [5.0])
    observations[column] = "too hot"

    with pytest.raises(ValueError, match="must contain numeric values"):
        assign_heatwave_proxy_labels(observations)


def test_custom_columns_and_thresholds_are_supported() -> None:
    observations = pd.DataFrame(
        {
            "observed_on": ["2024-05-01"],
            "daily_high": [36.0],
            "departure": [3.0],
        }
    )

    result = assign_heatwave_proxy_labels(
        observations,
        date_column="observed_on",
        max_temperature_column="daily_high",
        departure_column="departure",
        label_column="proxy_label",
        max_temperature_threshold_c=36.0,
        departure_threshold_c=3.0,
        minimum_consecutive_days=1,
    )

    assert result["proxy_label"].tolist() == [True]
    assert "heatwave_proxy" not in result.columns


def test_input_dataframe_is_not_modified() -> None:
    observations = _observations(
        ["2024-05-01", "2024-05-02"],
        [37.0, 38.0],
        [4.5, 5.0],
    )
    original = observations.copy(deep=True)

    result = assign_heatwave_proxy_labels(observations)

    pd.testing.assert_frame_equal(observations, original)
    assert "heatwave_proxy" not in observations.columns
    assert "heatwave_proxy" in result.columns