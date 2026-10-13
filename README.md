# Chennai Heatwave Early Warning System

A research prototype for exploring daily temperature extremes and a
one-day-ahead heatwave proxy classifier for Chennai, Tamil Nadu. The current
workflow accepts user-prepared NASA POWER daily historical data. It does not
download data automatically, query a live weather service, or provide
operational warnings.

> **Research prototype only.** The positive target is a simplified coastal
> heatwave proxy, not an official India Meteorological Department (IMD)
> declaration. Spatial requirements for official declarations are not
> implemented. For real alerts, follow IMD and local authorities.

## Current architecture

```text
User-prepared NASA POWER daily CSV
              |
              v
app/streamlit_app.py -- validates upload and exposes explicit controls
       |                         |
       v                         v
src/heatwave_labels.py     src/feature_engineering.py
       |                         |
       +------> src/train_model.py <------+
                         |
                         v
                 src/predict_warning.py
```

- `app/streamlit_app.py` validates a user-uploaded CSV, displays uploaded
  history, trains on button click, presents a chronological validation, and
  allows selected-date inference. It retains results in Streamlit session
  state so ordinary page reruns do not retrain.
- `src/heatwave_labels.py` assigns nullable research proxy labels. A known day
  qualifies when Tmax is at least 37 °C and departure from normal is at least
  4.5 °C. The default persistence is two consecutive calendar days.
- `src/feature_engineering.py` creates calendar-day lags, prior-calendar-day
  Tmax rolling means/maxima, month, and day of year. Missing days or readings
  remain missing; no weather values are fabricated.
- `src/train_model.py` uses those labels and predictors in a chronological
  split, fitting a median imputer, scaler, and class-balanced Logistic
  Regression pipeline on training rows only. It returns validation metrics
  and an in-memory model; it does not save a model.
- `src/predict_warning.py` uses the same feature builder for a selected date,
  checks the model feature schema, and excludes target-day readings from
  predictors.

`config/config.yaml` documents current code defaults for reference. The
dashboard does **not** load that file: its settings are the explicit sidebar
controls in `app/streamlit_app.py`. Change the app controls or code to change
runtime behavior; editing YAML alone has no effect.

## Dataset and CSV schema

The selected source remains **NASA POWER daily historical weather data for
Chennai**. The dashboard expects a CSV with one row per calendar day:

| Column | Required | Meaning |
|---|---:|---|
| `date` | Yes | Daily observation date, parseable as a calendar date |
| `max_temp_c` | Yes | Daily maximum temperature in °C |
| `departure_from_normal_c` | Yes for training | Tmax departure from normal in °C |
| `min_temp_c` | No | Daily minimum temperature; numeric when present |
| `humidity` | No | Daily humidity; numeric when present |
| `wind_speed` | No | Daily wind speed; numeric when present |

The date, maximum-temperature, and departure columns are required for the
dashboard's training workflow. Optional weather columns are used only when
present. Blank measurements are preserved as missing. The app rejects invalid
dates, duplicate calendar dates, and non-numeric weather values.

`departure_from_normal_c` must be computed before upload from a documented
climatological reference appropriate to the data, location, and period. The
current dashboard does not calculate a climatology, fetch reference normals,
or create departure values. A NASA POWER historical series by itself does not
automatically supply this project-specific departure field.

Check NASA POWER terms and attribution requirements before reuse or
publication. No live data is retrieved by this application.

## Windows setup

Open PowerShell in the repository root (`heatwave-early-warning`). The current
project environment has been exercised with Python 3.13.5; use a Python
version compatible with the pinned packages in `requirements.txt`.

Create the virtual environment and install the pinned dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell's execution policy prevents activation, continue to use
`.venv\Scripts\python.exe` directly in the commands below. `requirements.txt`
pins the project's Python dependencies, including pandas, scikit-learn,
Streamlit, and pytest.

## Launch the dashboard

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Upload a CSV matching the schema above. Confirm the explicit label thresholds,
persistence, train fraction, lags, and rolling windows in the sidebar, then
choose **Train and evaluate baseline**. After successful training, the
historical validation results and eligible prediction dates are shown. Select
a date and choose **Predict selected date** to run inference.

Training metrics are for the chronological validation period, not future
performance. Class `1` is shown as **Heatwave proxy warning**, not an official
IMD warning.

## Model workflow and metrics

The current trainer:

1. Builds proxy labels from daily Tmax and departure from normal.
2. Builds predictors from prior calendar-day weather lags, past-only Tmax
   rolling summaries, and target-date calendar features.
3. Splits sorted observations chronologically, using 80% for training by
   default and the later 20% for validation.
4. Treats missing labels as unknown and excludes them rather than treating
   them as negatives. A qualifying persistence run crossing the split is
   censored as unknown on both sides.
5. Fits median imputation and standardization only on training rows, followed
   by class-balanced Logistic Regression.
6. Reports precision, recall, F1, and a 2-by-2 confusion matrix. It also reports
   train/validation date ranges and warns when validation contains fewer than
   five positive cases by default.

Precision, recall, and F1 can be unreliable when the validation period contains
very few heatwave proxy positives. These scores are not model-performance
claims for real Chennai data; no agreed real dataset has yet been evaluated
here.

For selected-date inference, supply historical observations plus one row for
the target date. The target row must have a valid date; its weather values may
be blank. The predictor uses the trained feature schema and past observations
only. It rejects dates after the selected target date and requires sufficient
preceding calendar coverage for the configured feature windows.

**NASA POWER historical observations are not a live weather forecast.**
Predictions based on them are research extrapolations from past data and
calendar features. No numerical forecast feed is integrated.

## Run tests

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The tests use synthetic data to exercise labels, feature leakage and alignment,
chronological training, and prediction interfaces. They do not validate model
performance or the dashboard against a real NASA POWER CSV.

## Limitations and remaining work

- The real, agreed NASA POWER dataset has not been validated through the full
  dashboard workflow.
- The reference climatology and reproducible calculation of
  `departure_from_normal_c` must be agreed and documented upstream.
- The simplified label does not constitute an official IMD heatwave
  declaration and omits spatial declaration requirements.
- Rare positive labels and short validation periods can make metrics unstable.
- Missing weather values are not fabricated, but the training pipeline uses
  median imputation fitted only on the training partition. This is model
  preprocessing, not modification of uploaded source data.
- No live forecast, calibrated operational warning threshold, model persistence,
  spatial warning layer, or production alert service is implemented.
- EVT analysis and return levels remain separate planned research work; the
  dashboard's uploaded-record temperature summaries are descriptive only.

## Project structure

```text
heatwave-early-warning/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml              # Reference defaults; not loaded by app
├── app/
│   └── streamlit_app.py
├── src/
│   ├── heatwave_labels.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict_warning.py
└── tests/
    ├── test_project.py
    ├── test_feature_engineering.py
    ├── test_train_model.py
    └── test_predict_warning.py
```

Other source modules are reserved for related project work and are not needed
to launch the current dashboard workflow.
