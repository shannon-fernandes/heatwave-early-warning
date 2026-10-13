"""Synthetic tests for chronological model training."""

import pandas as pd
import pytest

from src.train_model import train_baseline_model


def _training_observations() -> pd.DataFrame:
    """Build deterministic weather with positive runs on both sides of a split."""
    hot_days = {2, 3, 11, 12, 15, 16}
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-04-01", periods=20, freq="D"),
            "max_temp_c": [
                38.0 if day in hot_days else 32.0 + day % 3
                for day in range(20)
            ],
            "departure_from_normal_c": [
                5.0 if day in hot_days else 2.0 for day in range(20)
            ],
        }
    )


def test_persistence_run_crossing_split_is_censored_in_both_partitions() -> None:
    result = train_baseline_model(
        _training_observations(),
        train_fraction=0.6,
        lag_days=(1,),
        rolling_windows=(2,),
        minimum_validation_positive_cases=1,
    )

    # The split is after row 11. The qualifying run on rows 11-12 straddles it,
    # so neither its training nor validation labels may use cross-split data.
    assert result.boundary_censored_labels == 2
    assert result.unknown_training_labels == 1
    assert result.unknown_validation_labels == 1
    assert result.training_rows == 11
    assert result.validation_rows == 7
    assert result.training_end < result.validation_start
    assert result.validation_positive_cases == 2


def test_unsorted_source_keeps_targets_aligned_after_chronological_sort() -> None:
    ordered = _training_observations()
    shuffled = ordered.sample(frac=1.0, random_state=7)
    options = {
        "train_fraction": 0.6,
        "lag_days": (1,),
        "rolling_windows": (2,),
        "minimum_validation_positive_cases": 1,
    }

    ordered_result = train_baseline_model(ordered, **options)
    shuffled_result = train_baseline_model(shuffled, **options)

    assert shuffled_result.training_rows == ordered_result.training_rows
    assert shuffled_result.validation_rows == ordered_result.validation_rows
    assert shuffled_result.unknown_training_labels == ordered_result.unknown_training_labels
    assert shuffled_result.unknown_validation_labels == ordered_result.unknown_validation_labels
    assert shuffled_result.boundary_censored_labels == ordered_result.boundary_censored_labels
    assert shuffled_result.training_start == ordered_result.training_start
    assert shuffled_result.training_end == ordered_result.training_end
    assert shuffled_result.validation_start == ordered_result.validation_start
    assert shuffled_result.validation_end == ordered_result.validation_end
    pd.testing.assert_frame_equal(
        shuffled_result.confusion_matrix,
        ordered_result.confusion_matrix,
    )


def test_empty_dataset_raises_clear_validation_error() -> None:
    observations = pd.DataFrame(
        columns=["date", "max_temp_c", "departure_from_normal_c"]
    )

    with pytest.raises(ValueError, match="at least one daily observation"):
        train_baseline_model(observations)


def test_single_class_training_partition_raises_clear_error() -> None:
    observations = _training_observations()
    observations["max_temp_c"] = 32.0
    observations["departure_from_normal_c"] = 2.0

    with pytest.raises(ValueError, match="both negative and positive"):
        train_baseline_model(
            observations,
            train_fraction=0.7,
            lag_days=(1,),
            rolling_windows=(2,),
        )
