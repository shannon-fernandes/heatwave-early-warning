"""Streamlit dashboard for the Chennai heatwave early warning research prototype."""

from __future__ import annotations

import hashlib
import math
from io import BytesIO
from typing import Any, Optional

import pandas as pd
import streamlit as st

from src.predict_warning import WarningPrediction, predict_warning
from src.train_model import TrainingResult, train_baseline_model


DATE_COLUMN = "date"
MAX_TEMPERATURE_COLUMN = "max_temp_c"
MIN_TEMPERATURE_COLUMN = "min_temp_c"
DEPARTURE_COLUMN = "departure_from_normal_c"
HUMIDITY_COLUMN = "humidity"
WIND_SPEED_COLUMN = "wind_speed"
OPTIONAL_WEATHER_COLUMNS = (
    MIN_TEMPERATURE_COLUMN,
    HUMIDITY_COLUMN,
    WIND_SPEED_COLUMN,
)
REQUIRED_TRAINING_COLUMNS = (
    DATE_COLUMN,
    MAX_TEMPERATURE_COLUMN,
    DEPARTURE_COLUMN,
)


def parse_weather_csv(contents: bytes) -> pd.DataFrame:
    """Read and validate a user-provided daily weather CSV.

    The input needs ``date``, ``max_temp_c``, and
    ``departure_from_normal_c`` columns to train. Optional supported weather
    columns are ``min_temp_c``, ``humidity``, and ``wind_speed``. Missing
    measurement values are retained as missing; no observations or
    departure-from-normal values are imputed or calculated here.
    """
    try:
        frame = pd.read_csv(BytesIO(contents))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc

    missing_columns = [
        column for column in REQUIRED_TRAINING_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "CSV is missing required training column(s): "
            f"{', '.join(missing_columns)}."
        )
    if frame.empty:
        raise ValueError("CSV contains no daily observations.")

    try:
        parsed_dates = pd.to_datetime(
            frame[DATE_COLUMN],
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse CSV date column {DATE_COLUMN!r}: {exc}") from exc
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        rows = (frame.index[invalid_dates] + 2).tolist()[:5]
        raise ValueError(
            f"Column {DATE_COLUMN!r} has invalid or missing dates on CSV row(s) {rows}."
        )
    normalized_dates = parsed_dates.dt.normalize()
    duplicate_dates = normalized_dates.duplicated(keep=False)
    if duplicate_dates.any():
        duplicate_values = (
            normalized_dates[duplicate_dates].drop_duplicates().astype(str).tolist()[:5]
        )
        raise ValueError(
            "CSV must contain at most one row per calendar day; duplicate date(s): "
            f"{', '.join(duplicate_values)}."
        )
    frame[DATE_COLUMN] = normalized_dates

    weather_columns = [
        MAX_TEMPERATURE_COLUMN,
        DEPARTURE_COLUMN,
        *[column for column in OPTIONAL_WEATHER_COLUMNS if column in frame.columns],
    ]
    for column in weather_columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Weather column {column!r} must contain numeric values or blanks."
            ) from exc
        finite_values = values.dropna()
        if not finite_values.map(math.isfinite).all():
            raise ValueError(f"Weather column {column!r} contains a non-finite value.")
        frame[column] = values

    return frame.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def _model_settings() -> dict[str, Any]:
    """Collect explicit model settings; config/config.yaml is currently empty."""
    st.sidebar.header("Model configuration")
    st.sidebar.caption(
        "These settings are explicit dashboard controls; no values are read from "
        "the currently empty config/config.yaml."
    )
    train_fraction = st.sidebar.slider(
        "Training fraction (chronological)",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05,
    )
    max_threshold = st.sidebar.number_input(
        "Proxy Tmax threshold (°C)", value=37.0, step=0.5
    )
    departure_threshold = st.sidebar.number_input(
        "Proxy departure threshold (°C)", value=4.5, step=0.5
    )
    persistence_days = st.sidebar.number_input(
        "Consecutive qualifying days", min_value=1, max_value=10, value=2
    )
    lag_days = tuple(
        st.sidebar.multiselect(
            "Prior calendar-day lags",
            options=[1, 2, 3, 5, 7, 14],
            default=[1, 2, 3, 7],
        )
    )
    rolling_windows = tuple(
        st.sidebar.multiselect(
            "Past Tmax rolling windows (days)",
            options=[2, 3, 5, 7, 14],
            default=[3, 7],
        )
    )
    st.sidebar.caption(
        "The default labels use a simplified IMD-style coastal proxy, not an "
        "official IMD declaration. Reference climatology is not calculated here."
    )
    return {
        "date_column": DATE_COLUMN,
        "max_temperature_column": MAX_TEMPERATURE_COLUMN,
        "departure_column": DEPARTURE_COLUMN,
        "label_column": "heatwave_proxy",
        "min_temperature_column": MIN_TEMPERATURE_COLUMN,
        "humidity_column": HUMIDITY_COLUMN,
        "wind_speed_column": WIND_SPEED_COLUMN,
        "max_temperature_threshold_c": float(max_threshold),
        "departure_threshold_c": float(departure_threshold),
        "minimum_consecutive_days": int(persistence_days),
        "lag_days": lag_days,
        "rolling_windows": rolling_windows,
        "train_fraction": float(train_fraction),
    }


def _settings_key(settings: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Create an immutable signature for the active training configuration."""
    return tuple(
        sorted(
            (
                name,
                tuple(value) if isinstance(value, (tuple, list)) else value,
            )
            for name, value in settings.items()
        )
    )


def _validate_dataset_upload(uploaded_file: Any) -> None:
    """Load a changed upload once and invalidate results tied to older data."""
    if uploaded_file is None:
        if st.session_state.get("upload_was_present", False):
            for key in (
                "dataset_signature",
                "weather_data",
                "weather_data_error",
                "training_result",
                "training_signature",
                "warning_prediction",
                "prediction_signature",
            ):
                st.session_state.pop(key, None)
        st.session_state["upload_was_present"] = False
        return

    st.session_state["upload_was_present"] = True
    contents = uploaded_file.getvalue()
    signature = hashlib.sha256(contents).hexdigest()
    if signature == st.session_state.get("dataset_signature"):
        return

    st.session_state["dataset_signature"] = signature
    st.session_state.pop("weather_data", None)
    st.session_state.pop("weather_data_error", None)
    st.session_state.pop("training_result", None)
    st.session_state.pop("training_signature", None)
    st.session_state.pop("warning_prediction", None)
    st.session_state.pop("prediction_signature", None)
    try:
        st.session_state["weather_data"] = parse_weather_csv(contents)
    except ValueError as exc:
        st.session_state["weather_data_error"] = str(exc)


def _render_upload() -> Optional[pd.DataFrame]:
    """Render upload guidance and return the validated dataset, if available."""
    st.subheader("Historical data")
    st.markdown(
        "Upload a **daily NASA POWER historical CSV**. Required columns: "
        "`date`, `max_temp_c`, and `departure_from_normal_c` (°C). Optional "
        "columns: `min_temp_c`, `humidity`, and `wind_speed`."
    )
    st.caption(
        "`departure_from_normal_c` must already be computed from a documented "
        "reference climatology. This dashboard does not derive it, fetch data, "
        "or fill missing source measurements."
    )
    uploaded_file = st.file_uploader(
        "Choose a historical weather CSV",
        type=["csv"],
        key="weather_csv_upload",
    )
    _validate_dataset_upload(uploaded_file)

    error = st.session_state.get("weather_data_error")
    if error:
        st.error(f"Dataset could not be loaded. {error}")
        return None

    dataset = st.session_state.get("weather_data")
    if dataset is None:
        st.info("Upload a validated daily CSV to begin.")
        return None

    missing_counts = dataset[
        [
            column
            for column in (
                MAX_TEMPERATURE_COLUMN,
                DEPARTURE_COLUMN,
                *OPTIONAL_WEATHER_COLUMNS,
            )
            if column in dataset.columns
        ]
    ].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        st.warning(
            "Missing source measurements are retained as missing. Counts by "
            "column: "
            + ", ".join(f"{column}: {count}" for column, count in missing_counts.items())
        )
    st.success(
        f"Loaded {len(dataset):,} daily rows from "
        f"{dataset[DATE_COLUMN].min():%Y-%m-%d} through "
        f"{dataset[DATE_COLUMN].max():%Y-%m-%d}."
    )
    return dataset


def _training_result(
    dataset: pd.DataFrame,
    settings: dict[str, Any],
) -> Optional[TrainingResult]:
    """Train only on button click and retain the result for ordinary reruns."""
    signature = (
        st.session_state.get("dataset_signature"),
        _settings_key(settings),
    )
    if st.session_state.get("training_signature") != signature:
        st.session_state.pop("training_result", None)
        st.session_state.pop("warning_prediction", None)
        st.session_state.pop("prediction_signature", None)

    st.subheader("Historical validation")
    st.caption(
        "This is a chronological holdout evaluation on uploaded historical "
        "observations. It is separate from the selected-date prediction below."
    )
    if st.button("Train and evaluate baseline", type="primary"):
        with st.spinner("Training on the chronological training period…"):
            try:
                result = train_baseline_model(dataset.copy(deep=True), **settings)
            except (TypeError, ValueError) as exc:
                st.session_state.pop("training_result", None)
                st.session_state.pop("training_signature", None)
                st.error(
                    f"Training could not be completed: {exc} Check the data, "
                    "label balance, selected thresholds, and split settings."
                )
            else:
                st.session_state["training_result"] = result
                st.session_state["training_signature"] = signature
                st.session_state.pop("warning_prediction", None)
                st.session_state.pop("prediction_signature", None)

    if st.session_state.get("training_signature") != signature:
        st.info("Train the baseline to view its historical validation results.")
        return None

    result = st.session_state.get("training_result")
    if not isinstance(result, TrainingResult):
        st.error("The saved training result is unavailable. Train the baseline again.")
        return None

    metrics = st.columns(3)
    for column, name in zip(metrics, ("precision", "recall", "f1")):
        column.metric(name.upper(), f"{result.metrics[name]:.3f}")
    st.markdown("**Confusion matrix** · rows are actual classes; columns are predicted classes.")
    st.dataframe(result.confusion_matrix, width="stretch")

    train_range, validation_range = st.columns(2)
    train_range.metric(
        "Training date range",
        f"{result.training_start:%Y-%m-%d} – {result.training_end:%Y-%m-%d}",
    )
    validation_range.metric(
        "Validation date range",
        f"{result.validation_start:%Y-%m-%d} – {result.validation_end:%Y-%m-%d}",
    )
    st.caption(
        f"Known labeled rows: {result.training_rows} training, "
        f"{result.validation_rows} validation. Unknown labels excluded: "
        f"{result.unknown_training_labels} training, "
        f"{result.unknown_validation_labels} validation; "
        f"{result.boundary_censored_labels} labels were censored at the split."
    )
    if result.validation_note:
        st.warning(result.validation_note)
    return result


def _candidate_prediction_dates(
    dataset: pd.DataFrame,
    result: TrainingResult,
    settings: dict[str, Any],
) -> list[pd.Timestamp]:
    """Return observed or next-day dates with a complete prior date window."""
    date_set = set(pd.DatetimeIndex(dataset[DATE_COLUMN]))
    max_lookback = max(
        [0, *settings["lag_days"], *settings["rolling_windows"]]
    )
    first_allowed = result.training_end.normalize() + pd.Timedelta(days=1)
    last_allowed = dataset[DATE_COLUMN].max() + pd.Timedelta(days=1)
    target_candidates = sorted(
        day for day in date_set if first_allowed <= day <= last_allowed
    )
    if last_allowed not in date_set:
        target_candidates.append(last_allowed)

    available: list[pd.Timestamp] = []
    for target in target_candidates:
        required_history = {
            target - pd.Timedelta(days=offset)
            for offset in range(1, max_lookback + 1)
        }
        if required_history.issubset(date_set):
            available.append(pd.Timestamp(target))
    return available


def _prediction_observations(
    dataset: pd.DataFrame,
    target: pd.Timestamp,
) -> pd.DataFrame:
    """Build the predictor input and explicitly withhold target-day readings."""
    history = dataset.loc[dataset[DATE_COLUMN] <= target].copy(deep=True)
    target_mask = history[DATE_COLUMN].eq(target)
    if not target_mask.any():
        target_row = {column: None for column in history.columns}
        target_row[DATE_COLUMN] = target
        history = pd.concat([history, pd.DataFrame([target_row])], ignore_index=True)
        target_mask = history[DATE_COLUMN].eq(target)

    for column in (
        MAX_TEMPERATURE_COLUMN,
        MIN_TEMPERATURE_COLUMN,
        DEPARTURE_COLUMN,
        HUMIDITY_COLUMN,
        WIND_SPEED_COLUMN,
        "heatwave_proxy",
    ):
        if column in history.columns:
            history.loc[target_mask, column] = pd.NA
    return history


def _render_prediction(
    dataset: pd.DataFrame,
    result: TrainingResult,
    settings: dict[str, Any],
) -> None:
    """Render a leakage-safe selected-date inference flow."""
    st.subheader("Selected-date prediction")
    st.caption(
        "This prediction is distinct from historical validation. The selected "
        "date’s weather and label are withheld from model inputs."
    )
    st.info(
        "NASA POWER daily historical observations are not a live forecast. "
        "A future-date result from these lagged observations is a research "
        "model extrapolation, not a real-time weather forecast or operational alert."
    )
    dates = _candidate_prediction_dates(dataset, result, settings)
    if not dates:
        st.warning(
            "No eligible prediction date is available. The dataset must contain "
            "a target date after the training period and uninterrupted preceding "
            f"calendar coverage for at least {max([0, *settings['lag_days'], *settings['rolling_windows']])} days."
        )
        return

    target = st.selectbox(
        "Prediction date",
        options=dates,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
        key="prediction_target_date",
    )
    prediction_signature = (
        st.session_state.get("dataset_signature"),
        st.session_state.get("training_signature"),
        pd.Timestamp(target),
    )
    if st.button("Predict selected date"):
        prediction_frame = _prediction_observations(dataset, pd.Timestamp(target))
        try:
            prediction = predict_warning(
                result,
                prediction_frame,
                pd.Timestamp(target),
                date_column=DATE_COLUMN,
                max_temperature_column=MAX_TEMPERATURE_COLUMN,
                min_temperature_column=MIN_TEMPERATURE_COLUMN,
                humidity_column=HUMIDITY_COLUMN,
                wind_speed_column=WIND_SPEED_COLUMN,
                lag_days=settings["lag_days"],
                rolling_windows=settings["rolling_windows"],
            )
        except (TypeError, ValueError) as exc:
            st.session_state.pop("warning_prediction", None)
            st.error(
                f"Prediction could not be produced: {exc} Check that the date "
                "and preceding observations match the trained feature schema."
            )
        else:
            st.session_state["warning_prediction"] = prediction
            st.session_state["prediction_signature"] = prediction_signature

    prediction = st.session_state.get("warning_prediction")
    if (
        st.session_state.get("prediction_signature") != prediction_signature
        or not isinstance(prediction, WarningPrediction)
    ):
        st.caption("Choose a date and select Predict to run inference.")
        return

    if prediction.target_date in set(dataset[DATE_COLUMN]):
        st.caption(
            "This is a historical holdout-date inference with target-day "
            "measurements withheld; the displayed validation metrics above "
            "remain the aggregate historical evaluation."
        )
    else:
        st.caption(
            "This target date is after the latest uploaded observation. "
            "No target-day weather readings were supplied."
        )
    if prediction.predicted_class == 1:
        st.warning("Heatwave proxy warning")
    else:
        st.success("No heatwave proxy warning predicted")
    if prediction.predicted_class_probability is not None:
        st.metric(
            "Predicted-class probability",
            f"{prediction.predicted_class_probability:.1%}",
        )
    st.caption(
        "This is a research proxy result, not an official IMD declaration or "
        "a live forecast. Follow IMD and local authorities for real alerts."
    )


def _render_history(dataset: pd.DataFrame) -> None:
    """Display validated uploaded observations without inventing measurements."""
    st.subheader("Latest uploaded weather observation")
    latest = dataset.iloc[-1]
    metric_columns = st.columns(3)
    for metric_column, (column, name, unit) in zip(
        metric_columns,
        (
            (MAX_TEMPERATURE_COLUMN, "Maximum temperature", " °C"),
            (MIN_TEMPERATURE_COLUMN, "Minimum temperature", " °C"),
            (HUMIDITY_COLUMN, "Relative humidity", " %"),
        ),
    ):
        value = latest.get(column)
        display_value = f"{value:.1f}{unit}" if pd.notna(value) else "—"
        metric_column.metric(name, display_value)
    st.caption(
        f"Latest observation in uploaded history: "
        f"{latest[DATE_COLUMN]:%Y-%m-%d}. This is not live weather."
    )

    st.subheader("Historical temperature trends")
    columns = [
        column
        for column in (MAX_TEMPERATURE_COLUMN, MIN_TEMPERATURE_COLUMN)
        if column in dataset.columns
    ]
    if not columns:
        st.info("Temperature history is unavailable in this CSV.")
        return
    chart = dataset.set_index(DATE_COLUMN)[columns]
    if chart.dropna(how="all").empty:
        st.info("The CSV contains no temperature measurements to plot.")
        return
    st.line_chart(chart, width="stretch")
    st.caption("Uploaded daily observations in °C. Missing values remain unfilled.")

    st.subheader("Observed extreme-temperature statistics")
    valid_maximum = dataset.dropna(subset=[MAX_TEMPERATURE_COLUMN])
    if valid_maximum.empty:
        st.info("No maximum-temperature readings are available for this summary.")
        return
    maximum_row = valid_maximum[MAX_TEMPERATURE_COLUMN].idxmax()
    high_value = valid_maximum.loc[maximum_row, MAX_TEMPERATURE_COLUMN]
    high_date = valid_maximum.loc[maximum_row, DATE_COLUMN]
    observed, dates_count = st.columns(2)
    observed.metric("Highest Tmax in uploaded record", f"{high_value:.1f} °C")
    dates_count.metric("Daily Tmax observations", f"{len(valid_maximum):,}")
    st.caption(
        f"Observed on {high_date:%Y-%m-%d}; descriptive statistics from this "
        "uploaded file only, not return levels or a fitted extreme-value model."
    )
    annual_maxima = (
        valid_maximum.assign(year=valid_maximum[DATE_COLUMN].dt.year)
        .groupby("year", as_index=True)[MAX_TEMPERATURE_COLUMN]
        .max()
        .rename("Annual observed Tmax maximum (°C)")
    )
    st.dataframe(annual_maxima, width="stretch")


def main() -> None:
    """Render the dashboard and its user-provided historical-data workflow."""
    st.set_page_config(
        page_title="Chennai Heatwave Early Warning",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
        .eyebrow {color: #587080; font-size: .78rem; font-weight: 700;
                  letter-spacing: .12em; text-transform: uppercase;}
        .status-pill {display: inline-block; border: 1px solid #edc97a;
                      border-radius: 999px; background: #fff8e6; color: #755516;
                      font-size: .82rem; font-weight: 650; padding: .35rem .7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="eyebrow">Heatwave &amp; extreme temperature monitoring</p>',
        unsafe_allow_html=True,
    )
    title, status = st.columns([3, 1])
    with title:
        st.title("Chennai, Tamil Nadu")
        st.caption("Early warning dashboard · Research prototype")
    with status:
        st.markdown(
            '<div style="text-align:right; padding-top:1.1rem">'
            '<span class="status-pill">Prototype · user-provided data only</span></div>',
            unsafe_allow_html=True,
        )

    settings = _model_settings()
    st.divider()
    dataset = _render_upload()
    if dataset is None:
        st.info(
            "No connected dataset is assumed. The project configuration is "
            "empty; thresholds and feature windows above are shown explicitly."
        )
        return

    st.divider()
    _render_history(dataset)

    st.divider()
    result = _training_result(dataset, settings)
    if result is not None:
        st.divider()
        _render_prediction(dataset, result, settings)

    st.divider()
    st.subheader("Methodology and limitations")
    st.markdown(
        """
        - Training labels use the configured Tmax and departure thresholds,
          plus the configured consecutive-day persistence rule. This is a
          simplified coastal research proxy and omits spatial requirements
          for official IMD declarations.
        - `departure_from_normal_c` must be prepared upstream using a documented
          reference climatology. This app will not infer a climatology or
          synthesize departure values.
        - Validation is chronological. Precision, recall, F1, and the confusion
          matrix describe the uploaded dataset's holdout period only; scores can
          be unstable when positive cases are few.
        - NASA POWER historical observations do not provide a live forecast.
          Prediction uses past observed weather and calendar features only;
          no operational forecast source is integrated.
        """
    )


if __name__ == "__main__":
    main()
