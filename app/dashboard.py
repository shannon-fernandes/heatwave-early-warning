"""A friendlier dashboard for the Chennai heatwave early-warning prototype.

Run from the repository root:   python -m streamlit run app/dashboard.py

Differences from app/streamlit_app.py: built-in data choices (no upload needed), five tabs
instead of one long page, plain-English explanations next to every number, a colour-coded
risk badge, and readable names for the model's drivers. The model code is the same.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.evt_panel import render_return_level_panel  # noqa: E402
from src.predict_warning import predict_warning  # noqa: E402
from src.train_model import TrainingResult, train_baseline_model  # noqa: E402
from src.warning_logic import risk_level  # noqa: E402

DATE, TMAX, TMIN, DEP, HUM, WIND = "date", "max_temp_c", "min_temp_c", "departure_from_normal_c", "humidity", "wind_speed"
OPTIONAL = (TMIN, HUM, WIND)
ACCENT, GREEN, AMBER, RED = "#E4572E", "#2E7D32", "#EF6C00", "#C62828"
DATA_CHOICES = {
    "Built-in sample (1981 to 2020, IMD-based)": ROOT / "data" / "sample" / "chennai_dashboard_sample.csv",
    "Live data (up to the latest published day)": ROOT / "data" / "processed" / "chennai_dashboard_live.csv",
    "Upload my own CSV": None,
}


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def parse_weather_csv(contents: bytes) -> pd.DataFrame:
    """Read and validate a daily weather CSV (date, max_temp_c, departure_from_normal_c + optional columns)."""
    try:
        frame = pd.read_csv(BytesIO(contents))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc
    missing = [c for c in (DATE, TMAX, DEP) if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("CSV contains no daily observations.")
    dates = pd.to_datetime(frame[DATE], errors="coerce", format="mixed")
    if dates.isna().any():
        rows = (frame.index[dates.isna()] + 2).tolist()[:5]
        raise ValueError(f"Column 'date' has invalid or missing dates on CSV row(s) {rows}.")
    dates = dates.dt.normalize()
    if dates.duplicated(keep=False).any():
        raise ValueError("CSV must contain at most one row per calendar day (duplicate dates found).")
    frame[DATE] = dates
    for column in [TMAX, DEP, *[c for c in OPTIONAL if c in frame.columns]]:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values or blanks.") from exc
        if not values.dropna().map(math.isfinite).all():
            raise ValueError(f"Column {column!r} contains a non-finite value.")
        frame[column] = values
    return frame.sort_values(DATE, kind="stable").reset_index(drop=True)


def load_dataset(choice: str, upload) -> tuple[Optional[pd.DataFrame], str]:
    """Return (dataset, message). The message explains why no dataset could be loaded."""
    path = DATA_CHOICES[choice]
    if path is None:
        if upload is None:
            return None, "Upload a daily CSV in the sidebar to begin."
        contents = upload.getvalue()
    else:
        if not path.exists():
            hint = "python -m src.update_live" if "live" in path.name else "python -m src.export_dashboard_csv"
            return None, f"{path.name} does not exist yet. Create it with `{hint}`, or pick another data option."
        contents = path.read_bytes()
    try:
        return parse_weather_csv(contents), ""
    except ValueError as exc:
        return None, f"This file could not be used. {exc}"


# ------------------------------------------------------------- settings
def model_settings() -> dict[str, Any]:
    with st.sidebar.expander("Advanced model settings", expanded=False):
        st.caption("The defaults match the project report. Changing them retrains the model.")
        train_fraction = st.slider("Share of history used for training", 0.5, 0.9, 0.8, 0.05,
                                   help="The earliest part of the record trains the model; the rest tests it.")
        tmax = st.number_input("Severe-day temperature (°C)", value=38.0, step=0.5)
        dep = st.number_input("Severe-day departure from normal (°C)", value=3.5, step=0.5)
        days = st.number_input("Consecutive days required", 1, 10, 2)
        lags = tuple(st.multiselect("Previous days used as inputs", [1, 2, 3, 5, 7, 14], default=[1, 2, 3, 7]))
        windows = tuple(st.multiselect("Rolling-average windows (days)", [2, 3, 5, 7, 14], default=[3, 7]))
    return {
        "date_column": DATE, "max_temperature_column": TMAX, "departure_column": DEP, "label_column": "heatwave_proxy",
        "min_temperature_column": TMIN, "humidity_column": HUM, "wind_speed_column": WIND,
        "max_temperature_threshold_c": float(tmax), "departure_threshold_c": float(dep),
        "minimum_consecutive_days": int(days), "lag_days": lags, "rolling_windows": windows,
        "train_fraction": float(train_fraction), "use_departure_features": True,
    }


def settings_key(settings: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in settings.items()))


# ------------------------------------------------------------- helpers
def show_chart(chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                                   # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


def show_table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, width="stretch")
    except TypeError:
        st.dataframe(frame, use_container_width=True)


def pretty_feature(name: str) -> str:
    base = {TMAX: "max temperature", TMIN: "min temperature", HUM: "humidity", WIND: "wind speed",
            DEP: "departure from normal"}
    m = re.fullmatch(r"(.+)_lag_(\d+)", name)
    if m:
        k = int(m.group(2))
        label = base.get(m.group(1), m.group(1)).capitalize()
        return f"{label}, {'yesterday' if k == 1 else f'{k} days ago'}"
    m = re.fullmatch(rf"{TMAX}_rolling_(mean|max)_(\d+)", name)
    if m:
        return f"{'Average' if m.group(1) == 'mean' else 'Highest'} max temperature, previous {m.group(2)} days"
    return {"month": "Month of year", "day_of_year": "Day of year"}.get(name, name)


def candidate_dates(dataset: pd.DataFrame, result: TrainingResult, settings: dict) -> list[pd.Timestamp]:
    """Dates after the training period (plus the day after the data ends) with a complete look-back window."""
    have = set(pd.DatetimeIndex(dataset[DATE]))
    lookback = max([0, *settings["lag_days"], *settings["rolling_windows"]])
    first = result.training_end.normalize() + pd.Timedelta(days=1)
    last = dataset[DATE].max() + pd.Timedelta(days=1)
    targets = sorted(d for d in have if first <= d <= last)
    if last not in have:
        targets.append(last)
    return [pd.Timestamp(t) for t in targets
            if {t - pd.Timedelta(days=k) for k in range(1, lookback + 1)} <= have]


def prediction_frame(dataset: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """History up to the target date, with the target day's own readings withheld."""
    hist = dataset.loc[dataset[DATE] <= target].copy(deep=True)
    mask = hist[DATE].eq(target)
    if not mask.any():
        row = {c: None for c in hist.columns}
        row[DATE] = target
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        mask = hist[DATE].eq(target)
    for column in (TMAX, TMIN, DEP, HUM, WIND, "heatwave_proxy"):
        if column in hist.columns:
            hist.loc[mask, column] = pd.NA
    return hist


def risk_badge(level: str) -> None:
    colour = {"Low": GREEN, "Moderate": AMBER, "High": RED}[level]
    st.markdown(
        f"<div style='background:{colour};color:white;padding:1.1rem 1.4rem;border-radius:14px;"
        f"font-size:2.1rem;font-weight:700;line-height:1.1'>{level} risk</div>", unsafe_allow_html=True)


RISK_TEXT = {
    "Low": "Little sign of a severe hot day. Conditions do not resemble the build-up to past hot spells.",
    "Moderate": "Conditions resemble the start of past severe hot spells. Worth keeping an eye on.",
    "High": "Conditions strongly resemble past severe hot-day spells.",
}


# ----------------------------------------------------------------- tabs
def tab_overview(dataset: pd.DataFrame, settings: dict) -> None:
    valid = dataset.dropna(subset=[TMAX])
    latest = valid.iloc[-1]
    record = valid.loc[valid[TMAX].idxmax()]
    recent = dataset[dataset[DATE] > dataset[DATE].max() - pd.DateOffset(years=1)]
    hot_days = int((recent[TMAX] >= settings["max_temperature_threshold_c"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest observation", f"{latest[DATE]:%d %b %Y}", help="The newest day with a temperature reading in this file. This is not live weather.")
    c2.metric("Max temperature that day", f"{latest[TMAX]:.1f} °C")
    c3.metric("Hottest day in this file", f"{record[TMAX]:.1f} °C", f"{record[DATE]:%d %b %Y}", delta_color="off")
    c4.metric(f"Days at or above {settings['max_temperature_threshold_c']:.0f} °C", hot_days, "in the last 12 months", delta_color="off")

    st.subheader("Daily maximum temperature")
    years = st.select_slider("Time window", options=[1, 2, 3, 5, 10, 40], value=3,
                             format_func=lambda y: f"last {y} year{'s' if y > 1 else ''}", key="overview_years")
    window = valid[valid[DATE] > valid[DATE].max() - pd.DateOffset(years=years)]
    line = alt.Chart(window).mark_line(color=ACCENT, opacity=0.85).encode(
        x=alt.X("date:T", title=None), y=alt.Y(f"{TMAX}:Q", title="Daily maximum (°C)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip(f"{TMAX}:Q", title="Max °C", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [settings["max_temperature_threshold_c"]]})).mark_rule(
        strokeDash=[6, 4], color="#444").encode(y="y:Q")
    show_chart(line + rule)
    st.caption(f"The dashed line is the {settings['max_temperature_threshold_c']:.0f} °C severe-day temperature.")

    st.subheader("Is the hottest day of the year getting hotter?")
    by_year = valid.assign(year=valid[DATE].dt.year).groupby("year")[TMAX].agg(["max", "count"])
    by_year = by_year[by_year["count"] >= 300].reset_index().rename(columns={"max": "annual_max"})
    if len(by_year) >= 5:
        dots = alt.Chart(by_year).mark_circle(size=70, color=ACCENT).encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
            y=alt.Y("annual_max:Q", title="Hottest day (°C)", scale=alt.Scale(zero=False)),
            tooltip=["year:Q", alt.Tooltip("annual_max:Q", format=".1f")])
        trend = dots.transform_regression("year", "annual_max").mark_line(color="#444")
        show_chart(dots + trend)
        st.caption("Each dot is the hottest day of a full year; the line is the overall trend in this file.")
    else:
        st.info("Not enough full years in this file to draw a yearly trend.")


def tab_extremes(dataset: pd.DataFrame) -> None:
    st.markdown("**How rare is an extreme temperature?** This uses *Extreme Value Theory* on the hottest day of each "
                "March to June season. It describes the long run and is **not a forecast**.")
    render_return_level_panel(dataset)
    with st.expander("How to read the return-level table"):
        st.markdown(
            "- A **10-year return level of 43.2 °C** means the season's hottest day exceeds 43.2 °C about once every ten years.\n"
            "- The **95% interval** shows how uncertain that estimate is. Longer return periods are less certain.\n"
            "- If the **stationarity check** reports a clear trend, today's extremes are likely hotter than the table suggests.\n"
            "- Results depend on the data source: IMD-based and NASA POWER files give different levels, so do not mix them.")


def tab_model(dataset: pd.DataFrame, settings: dict, data_key: str) -> Optional[TrainingResult]:
    st.markdown("Train the model on the earlier part of the record, then test it on the later part it has never seen.")
    signature = (data_key, settings_key(settings))
    if st.button("Train the model", type="primary", key="train_btn"):
        with st.spinner("Training on the earlier history..."):
            try:
                result = train_baseline_model(dataset.copy(deep=True), **settings)
            except (TypeError, ValueError) as exc:
                st.session_state.pop("result", None)
                st.error(f"Training could not be completed: {exc} Try other thresholds or more data.")
            else:
                st.session_state["result"], st.session_state["signature"] = result, signature
                st.session_state.pop("prediction", None)
    if st.session_state.get("signature") != signature or "result" not in st.session_state:
        st.info("Press **Train the model** to see how well it performs on past data.")
        return None
    result: TrainingResult = st.session_state["result"]
    cm = result.confusion_matrix
    tn, fp, fn, tp = int(cm.loc[0, 0]), int(cm.loc[0, 1]), int(cm.loc[1, 0]), int(cm.loc[1, 1])
    events = tp + fn
    st.success(f"Test period: {result.validation_start:%d %b %Y} to {result.validation_end:%d %b %Y}. "
               f"It contained **{events}** real severe hot days. The model caught **{tp}** and missed **{fn}**, "
               f"and raised **{fp}** false alarms.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Events caught (recall)", f"{result.metrics['recall']:.0%}", help="Of the real severe hot days, the share the model warned about.")
    c2.metric("Warnings that were right (precision)", f"{result.metrics['precision']:.0%}", help="Of all the days the model warned about, the share that really were severe.")
    c3.metric("Balance of the two (F1)", f"{result.metrics['f1']:.2f}", help="Combines the two scores. 1 is perfect, 0 is useless.")
    table = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually: no severe day", "Actually: severe day"],
                         columns=["Model said: no warning", "Model said: warning"])
    show_table(table)
    st.caption(f"Scores use the plain 50% cut-off. The risk level on the next tab uses a stricter, cost-based threshold "
               f"({result.decision_threshold:.0%}), so it warns less often than these numbers suggest.")
    if result.validation_note:
        st.warning(result.validation_note)
    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Few events:** severe hot days are rare, so every score here is uncertain. Treat them as a rough guide.\n"
            "- **Why not accuracy?** Over 99% of days are not events, so a model that always says 'no' would look very accurate.\n"
            "- **Honest comparison:** simply repeating yesterday's status ('persistence') is a strong baseline. "
            "See `python -m src.run_evaluation` for the full comparison.")
    return result


def tab_predict(dataset: pd.DataFrame, settings: dict, result: Optional[TrainingResult]) -> None:
    if result is None:
        st.info("Train the model on the **Model check** tab first.")
        return
    dates = candidate_dates(dataset, result, settings)
    if not dates:
        st.warning("No date is available to predict: the data needs a complete run of previous days after the training period.")
        return
    st.markdown("Pick a day. The model uses only the **days before it**; that day's own weather is hidden from the model.")
    target = st.selectbox("Day to assess", dates, index=len(dates) - 1, format_func=lambda d: d.strftime("%d %b %Y"), key="target_date")
    if st.button("Show risk for this day", type="primary", key="predict_btn"):
        try:
            st.session_state["prediction"] = (pd.Timestamp(target), predict_warning(
                result, prediction_frame(dataset, pd.Timestamp(target)), pd.Timestamp(target),
                lag_days=settings["lag_days"], rolling_windows=settings["rolling_windows"]))
        except (TypeError, ValueError) as exc:
            st.session_state.pop("prediction", None)
            st.error(f"Could not produce a prediction: {exc}")
    shown = st.session_state.get("prediction")
    if not shown or shown[0] != pd.Timestamp(target):
        return
    _, prediction = shown
    p = prediction.positive_class_probability
    if p is None or result.decision_threshold is None:
        st.warning("This model does not provide a heatwave probability.")
        return
    level = risk_level(p, result.decision_threshold)
    left, right = st.columns([1, 2])
    with left:
        risk_badge(level)
        st.metric("Chance of a severe hot day", f"{p:.1%}")
    with right:
        st.markdown(f"**{RISK_TEXT[level]}**")
        st.write(f"The model estimates a **{p:.1%}** chance that {target:%d %b %Y} is a severe hot day "
                 f"(at least {settings['max_temperature_threshold_c']:.0f} °C and {settings['departure_threshold_c']:.1f} °C above normal, "
                 f"for {settings['minimum_consecutive_days']} days running). A warning is raised at {result.decision_threshold:.0%} or more.")
        behind = (pd.Timestamp.today().normalize() - pd.Timestamp(target)).days
        if behind > 0:
            st.caption(f"This day is {behind} day(s) before today. The data file ends on {dataset[DATE].max():%d %b %Y}.")
    if prediction.contributing_factors:
        st.markdown("#### What is driving this?")
        drivers = pd.DataFrame([(pretty_feature(n), v) for n, v in prediction.contributing_factors], columns=["factor", "effect"])
        bars = alt.Chart(drivers).mark_bar().encode(
            x=alt.X("effect:Q", title="Pushes away from a warning   <   0   >   pushes towards a warning"),
            y=alt.Y("factor:N", sort=None, title=None),
            color=alt.condition(alt.datum.effect > 0, alt.value(RED), alt.value("#1F77B4")),
            tooltip=["factor:N", alt.Tooltip("effect:Q", format="+.2f")])
        show_chart(bars)
        st.caption("Red bars push towards a warning, blue bars push away from one. Longer bars matter more.")
    st.caption("Research prototype: not an official IMD declaration or a weather forecast. Follow IMD and local authorities for real alerts.")


def tab_guide() -> None:
    st.markdown("""
#### What is this?
A research prototype that combines two statistical tools for Chennai's extreme heat:

- **Extreme Value Theory** describes how rare extreme temperatures are and whether they are changing (long-term planning).
- **A machine-learning classifier** estimates the chance that the next day is a *severe hot day* (short-term warning).

The two are shown separately because they answer different questions.

#### Words used here
| Term | Meaning |
|---|---|
| Departure from normal | How many °C a day's maximum is above the usual value for that date (average of 1951 to 2005). |
| Severe hot-day proxy | A working definition: at least 38 °C and 3.5 °C above normal for 2 days running. It is **not** an official IMD heatwave. |
| Return level | The temperature exceeded on average once every N years. |
| Recall / precision | Share of real events caught / share of warnings that were right. |
| Risk level | Low below the warning threshold, Moderate above it, High halfway between the threshold and 100%. |

#### Important limits
- The model uses past observations and the calendar only. It is **not a weather forecast**.
- Severe hot days are rare, so test scores are uncertain.
- The latest day it can assess is the day after the last observation in the data file.
- This is a student research project, not an operational warning service.
""")


# ----------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Chennai Heatwave Early Warning", page_icon="🌡️", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1150px; padding-top: 1.6rem;}
    div[data-testid="stMetric"] {background:#FFF4EC; border-radius:12px; padding:.8rem 1rem;}
    .pill {display:inline-block; border:1px solid #F2C9B5; background:#FFF4EC; color:#9A3B1A; border-radius:999px;
           font-size:.8rem; font-weight:600; padding:.25rem .7rem;}
    </style>""", unsafe_allow_html=True)

    head, pill = st.columns([4, 1])
    head.title("Chennai Heatwave Early Warning")
    head.caption("How rare is extreme heat, and how likely is a severe hot day tomorrow?")
    pill.markdown("<div style='text-align:right;padding-top:1.6rem'><span class='pill'>Research prototype</span></div>", unsafe_allow_html=True)

    st.sidebar.header("Data")
    choice = st.sidebar.radio("Which data?", list(DATA_CHOICES), key="data_choice", label_visibility="collapsed")
    upload = st.sidebar.file_uploader("CSV file", type=["csv"], key="upload") if DATA_CHOICES[choice] is None else None
    settings = model_settings()

    dataset, problem = load_dataset(choice, upload)
    if dataset is None:
        st.info(problem)
        tab_guide()
        return
    data_key = hashlib.sha256(pd.util.hash_pandas_object(dataset, index=True).to_numpy().tobytes()).hexdigest()[:16]
    gaps = int(dataset[[TMAX, DEP, *[c for c in OPTIONAL if c in dataset.columns]]].isna().sum().sum())
    st.caption(f"{len(dataset):,} days loaded, {dataset[DATE].min():%d %b %Y} to {dataset[DATE].max():%d %b %Y}"
               + (f" ({gaps} missing values left as missing)." if gaps else "."))

    t_over, t_ext, t_model, t_pred, t_guide = st.tabs(["Overview", "How extreme?", "Model check", "Predict a day", "Guide"])
    with t_over:
        tab_overview(dataset, settings)
    with t_ext:
        tab_extremes(dataset)
    with t_model:
        result = tab_model(dataset, settings, data_key)
    with t_pred:
        tab_predict(dataset, settings, result)
    with t_guide:
        tab_guide()


if __name__ == "__main__":
    main()
