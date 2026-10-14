"""Synthetic prediction tests for the heatwave proxy warning pipeline."""

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.predict_warning import predict_warning
from src.train_model import TrainingResult, train_baseline_model


def _training_observations() -> pd.DataFrame:
    """Create synthetic daily observations with both label classes."""
    hot_days = {2, 3, 8, 9, 14, 15, 20, 21, 27, 28, 34, 35, 38, 39}
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-04-01", periods=40, freq="D"),
            "max_temp_c": [
                38.0 if day in hot_days else 32.0 + day % 4
                for day in range(40)
            ],
            "min_temp_c": [22.0 + day % 3 for day in range(40)],
            "humidity": [55.0 + day % 7 for day in range(40)],
            "wind_speed": [2.0 + day % 2 for day in range(40)],
            "departure_from_normal_c": [
                5.0 if day in hot_days else 2.0 for day in range(40)
            ],
        }
    )


@pytest.fixture
def trained_result() -> TrainingResult:
    """Fit the actual project training pipeline on synthetic observations."""
    return train_baseline_model(
        _training_observations(),
        train_fraction=0.75,
        lag_days=(1, 2),
        rolling_windows=(2,),
        minimum_validation_positive_cases=1,
    )


def _prediction_frame() -> pd.DataFrame:
    """Append a future target row with unknown weather to synthetic history."""
    observations = _training_observations().drop(columns="departure_from_normal_c")
    target_row = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-05-11")],
            "max_temp_c": [None],
            "min_temp_c": [None],
            "humidity": [None],
            "wind_speed": [None],
            "heatwave_proxy": [None],
        }
    )
    return pd.concat([observations, target_row], ignore_index=True)


class _RecordingEstimator:
    """Small fitted-estimator stand-in that records model input columns."""

    classes_ = np.array([0, 1])

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.received: Optional[pd.DataFrame] = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.received = features.copy()
        return np.array([1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.received = features.copy()
        return np.array([[0.2, 0.8]])


def _recording_result(result: TrainingResult) -> tuple[TrainingResult, _RecordingEstimator]:
    estimator = _RecordingEstimator(result.feature_columns)
    return replace(result, model=estimator), estimator


def test_target_weather_can_be_missing_and_probability_is_returned(
    trained_result: TrainingResult,
) -> None:
    prediction = predict_warning(
        trained_result,
        _prediction_frame(),
        "2024-05-11",
        lag_days=(1, 2),
        rolling_windows=(2,),
    )

    assert prediction.target_date == pd.Timestamp("2024-05-11")
    assert prediction.predicted_class in (0, 1)
    assert prediction.predicted_class_probability is not None
    assert 0.0 <= prediction.predicted_class_probability <= 1.0


def test_prediction_uses_training_feature_columns_in_exact_order(
    trained_result: TrainingResult,
) -> None:
    result_with_recorder, estimator = _recording_result(trained_result)

    predict_warning(
        result_with_recorder,
        _prediction_frame(),
        "2024-05-11",
        lag_days=(1, 2),
        rolling_windows=(2,),
    )

    assert estimator.received is not None
    assert tuple(estimator.received.columns) == trained_result.feature_columns
    assert estimator.feature_names_in_.tolist() == list(trained_result.feature_columns)


def test_prediction_selects_the_requested_date_even_when_input_is_unsorted(
    trained_result: TrainingResult,
) -> None:
    observations = _prediction_frame().iloc[::-1]
    result_with_recorder, estimator = _recording_result(trained_result)

    prediction = predict_warning(
        result_with_recorder,
        observations,
        pd.Timestamp("2024-05-11 18:00"),
        lag_days=(1, 2),
        rolling_windows=(2,),
    )

    assert prediction.target_date == pd.Timestamp("2024-05-11")
    assert estimator.received is not None
    assert estimator.received.shape[0] == 1
    assert estimator.received.iloc[0]["max_temp_c_lag_1"] == 38.0


def test_target_day_weather_does_not_leak_into_prediction_features(
    trained_result: TrainingResult,
) -> None:
    base = _prediction_frame()
    changed = base.copy()
    target_index = changed.index[-1]
    changed.loc[target_index, ["max_temp_c", "min_temp_c", "humidity", "wind_speed"]] = [
        999.0,
        888.0,
        777.0,
        666.0,
    ]
    first_result, first_estimator = _recording_result(trained_result)
    second_result, second_estimator = _recording_result(trained_result)

    predict_warning(
        first_result,
        base,
        "2024-05-11",
        lag_days=(1, 2),
        rolling_windows=(2,),
    )
    predict_warning(
        second_result,
        changed,
        "2024-05-11",
        lag_days=(1, 2),
        rolling_windows=(2,),
    )

    assert first_estimator.received is not None
    assert second_estimator.received is not None
    pd.testing.assert_frame_equal(
        first_estimator.received,
        second_estimator.received,
    )
    assert first_estimator.received.iloc[0]["max_temp_c_lag_1"] == 38.0
    assert "heatwave_proxy" not in first_estimator.received.columns


@pytest.mark.parametrize(
    ("dates", "target_date", "message"),
    [
        (
            ["2024-05-10", "2024-05-11", "2024-05-11 12:00"],
            "2024-05-11",
            "Duplicate daily dates",
        ),
        (
            ["2024-05-10", "not-a-date"],
            "2024-05-11",
            "invalid or missing dates",
        ),
    ],
    ids=["duplicate-dates", "invalid-date"],
)
def test_invalid_or_duplicate_dates_raise_clear_errors(
    trained_result: TrainingResult,
    dates: list[str],
    target_date: str,
    message: str,
) -> None:
    observations = pd.DataFrame(
        {
            "date": dates,
            "max_temp_c": [35.0] * len(dates),
            "min_temp_c": [None] * len(dates),
            "humidity": [None] * len(dates),
            "wind_speed": [None] * len(dates),
        }
    )

    with pytest.raises(ValueError, match=message):
        predict_warning(
            trained_result,
            observations,
            target_date,
            lag_days=(1, 2),
            rolling_windows=(2,),
        )


def test_prediction_requires_exactly_one_requested_target_date(
    trained_result: TrainingResult,
) -> None:
    observations = _prediction_frame().iloc[:-1]

    with pytest.raises(ValueError, match="Exactly one row matching target_date"):
        predict_warning(
            trained_result,
            observations,
            "2024-05-11",
            lag_days=(1, 2),
            rolling_windows=(2,),
        )


def test_prediction_rejects_dates_after_target_date(
    trained_result: TrainingResult,
) -> None:
    observations = _prediction_frame()
    later_row = observations.iloc[[-1]].copy()
    later_row["date"] = pd.Timestamp("2024-05-12")
    observations = pd.concat([observations, later_row], ignore_index=True)

    with pytest.raises(ValueError, match="dates after target_date"):
        predict_warning(
            trained_result,
            observations,
            "2024-05-11",
            lag_days=(1, 2),
            rolling_windows=(2,),
        )
