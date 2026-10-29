# Chennai Heatwave Early Warning System

A research prototype for exploring extreme heat in Chennai, Tamil Nadu. It
combines an Extreme Value Theory (EVT) analysis of seasonal temperature extremes
with a one-day-ahead classifier for a severe hot-day proxy.

> **Research prototype only.** The classifier target is a simplified proxy,
> not an official India Meteorological Department (IMD) heatwave declaration.
> The system is not a weather forecast or an operational warning service. For
> real alerts, follow IMD and local authorities.

## Dashboard

The main dashboard is a Streamlit app with five tabs:

- **Overview** — recent observations, daily maximum-temperature history, and
  annual maxima.
- **How extreme?** — March–June return levels, uncertainty intervals, and
  stationarity diagnostics, calculated separately from the classifier.
- **Model check** — train on the earlier part of the selected record and inspect
  performance on its later part.
- **Predict a day** — estimate the chance of a severe hot day using observations
  before the selected date, with a risk badge and contributing factors.
- **Guide** — definitions and interpretation notes.

Choose one of the built-in sample or live datasets in the sidebar, or upload a
CSV. The sample is available in a fresh checkout. The live option appears once
`src.update_live` has generated its CSV. The app also supports a user-provided
CSV with this schema:

| Column | Required | Meaning |
|---|---:|---|
| `date` | Yes | Daily observation date |
| `max_temp_c` | Yes | Daily maximum temperature in °C |
| `departure_from_normal_c` | Yes | Tmax departure from normal in °C |
| `min_temp_c` | No | Daily minimum temperature in °C |
| `humidity` | No | Daily humidity |
| `wind_speed` | No | Daily wind speed |

Dates must be valid and unique by calendar day. Weather columns must be numeric
when populated; blanks remain missing. The app does not calculate departures
from normal for uploaded data. Prepare that column using a documented
climatology appropriate to the source and period.

## Setup

Use Python and install the dependencies from the repository root.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the project's runtime and test dependencies.

## Run the dashboard

From the repository root:

```bash
python -m streamlit run app/dashboard.py
```

`app/streamlit_app.py` remains available as the simpler upload-first dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

The main dashboard uses the severe hot-day proxy by default: Tmax at least
38 °C and departure from normal at least 3.5 °C for two consecutive days.
Thresholds, persistence, training fraction, lag days, and rolling windows can be
changed in **Advanced model settings**. The target-day weather is withheld from
prediction features; predictions use the history and calendar features only.
NASA POWER observations are historical data, not a numerical weather forecast.

## Data pipeline and analysis

The configured historical workflow uses OpenCity's Chennai temperature record
and NASA POWER weather data. Run these commands in order:

```bash
python -m src.data_download
python -m src.data_preprocessing
python -m src.evt_analysis
python -m src.export_dashboard_csv
python -m src.run_evaluation
```

`src.data_download` accepts `--source opencity`, `--source nasa`, or
`--source all`; it skips existing downloads unless `--force` is provided.
Preprocessing validates ranges, preserves missing days rather than
interpolating, and writes data-quality and source-comparison reports.

The dashboard CSV export calculates departure from normal as Tmax minus the
day-of-year mean Tmax through the configured reference end year (currently
2005). This is a fixed reference period, not an official 30-year normal. The
machine-learning period is configured as 1981–2020 to avoid including the
OpenCity source change from grid data to a station record in 2021.

To generate a current NASA POWER dataset for the dashboard:

```bash
python -m src.update_live
python -m src.run_evaluation --input data/processed/chennai_dashboard_live.csv --tag live
```

NASA POWER data is published with a delay. The newest usable observation may
therefore be several days old, and the dashboard's latest prediction date is
the day after the last available observation. Live POWER results should be
reported separately from results based on the OpenCity-based historical
dataset because the sources have different temperature characteristics.

## Labels and evaluation

The evaluation reports two targets:

- **Severe hot-day proxy** (working target): Tmax ≥ 38 °C and departure from
  normal ≥ 3.5 °C for two consecutive days.
- **IMD-style proxy** (reported for transparency): Tmax ≥ 37 °C and departure
  from normal ≥ 4.5 °C for two consecutive days. It is not an official IMD
  declaration and yields very few events for Chennai.

`src.run_evaluation` uses a chronological train/validation split (80/20 by
default). It chooses model and climatology warning thresholds on the first half
of validation and scores on the second half, with persistence (yesterday's
label) as another baseline. Missing/unknown labels are not treated as negative
events. Because positive cases are rare, all reported performance estimates
should be treated as uncertain; persistence is a strong baseline.

## Generated results

Analysis outputs are written under `outputs/`:

- `outputs/tables/gev_parameters.csv` and `gev_return_levels.csv` contain EVT
  fit parameters and return levels. The primary fit uses the homogeneous
  1951–2020 period; a longer-period sensitivity result is also reported.
- `outputs/tables/model_comparison_*.csv` compares the classifier with
  climatology and persistence.
- `outputs/tables/calibration_*.csv` and `outputs/figures/calibration_*.png`
  report probability calibration.
- `outputs/figures/pr_curve_*.png`, `gev_*.png`, and
  `seasonal_maxima_trend.png` contain evaluation and EVT plots.
- `outputs/tables/label_counts*.csv`, `data_quality_report.csv`,
  `source_break_report.csv`, and `source_comparison.csv` document event counts
  and data checks.

Output files are generated by the commands above and are not required to launch
the sample dashboard.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest
```

The tests cover data validation, labels, feature generation, training,
prediction, evaluation, live-data handling, EVT analysis, and dashboard
behavior. They do not establish operational warning performance.

## Limitations

- The working severe-day target is a proxy, not an official IMD definition or
  spatially verified declaration.
- The fixed 1951–2005 departure climatology is a short reference period, not an
  official 30-year normal.
- The OpenCity temperature source changes from grid to station data in 2021.
  The ML analysis excludes those later years; EVT reports a primary
  homogeneous-period fit and a longer-period sensitivity analysis.
- NASA POWER is a gridded reanalysis with different temperature characteristics
  from the OpenCity/IMD record and is published with a delay.
- Few positive events make model scores and long return-period estimates
  uncertain. Cost-based warning thresholds use an assumed cost ratio, not
  measured public-health costs.
- Missing observations are not fabricated during preprocessing. The ML
  pipeline uses imputation fitted on training data; it does not change the
  source CSV.
- No numerical weather forecast feed, production alert service, or guarantee of
  calibrated operational probabilities is provided.
