# Heatwave and Extreme Temperature Early Warning System for India

Extreme Value Theory (GEV) and machine learning, with Chennai as the study area.

> **Research prototype.** This is an academic project. It is **not** an
> operational, official, or government-endorsed warning service. Heatwave labels
> are an **IMD-style proxy**, not official IMD declarations. Safety text is
> general guidance only. For real alerts, follow the India Meteorological
> Department (IMD) and local authorities.

## Problem statement

Heatwaves cause heat-related illness and deaths and strain power and water
systems. Early warning needs (a) an estimate of how rare an extreme temperature
is and (b) a forecast of heatwave days before they happen. This project combines
Extreme Value Theory for (a) with a machine learning classifier for (b).

## Research questions

1. **EVT:** What are the 2-, 5-, 10- and 20-year return levels of annual maximum
   temperature in Chennai, and how uncertain are they?
2. **Trend:** Is there evidence that extremes are changing, making a stationary
   GEV questionable?
3. **ML:** Can a leakage-safe, interpretable classifier forecast a heatwave day
   one day ahead better than climatology and persistence baselines?
4. **Warning:** Can calibrated probabilities and cost-aware thresholds be turned
   into understandable risk levels?

## Study area and period

- Location: Chennai, Tamil Nadu (about 13.08 N, 80.27 E), a coastal station
- Period: 1981 to the latest full year (EVT may use the longer IMD record)
- Both settings live in `config/config.yaml`, so the location and years can be
  changed without editing code

## Data sources

| Dataset | Role | Access |
|---|---|---|
| IMD gridded daily Tmax/Tmin | Primary: EVT and heatwave labels | Open download from IMD Pune; `imdlib` can automate it |
| NASA POWER daily data | Supplementary: humidity, dew point, wind | Free API, no key needed |
| ERA5-Land | Optional validation | Free Copernicus CDS account and API token |

Raw data is not stored in this repository. Check each provider's current terms
and attribution requirements before reusing or publishing results. Credentials
stay in `.env` or `~/.cdsapirc` and are never committed.

## Methodology

**Extreme Value Theory.** Annual block maxima of daily Tmax are fitted with a
Generalized Extreme Value distribution by maximum likelihood. Return levels are
reported with confidence intervals (profile likelihood or bootstrap).
Diagnostics include QQ/PP plots, return-level plots and a trend test on the
maxima. SciPy's `genextreme` uses shape `c = -xi`, and the code handles this
sign convention explicitly.

**Heatwave labels.** An IMD-style proxy for a coastal station: a day exceeds a
temperature floor and a day-of-year climatological threshold computed from the
training period only, with a persistence rule. Exact thresholds are documented
in `src/heatwave_labels.py` and checked against IMD's current criteria.

**Machine learning.** Features use only information available at forecast issue
time (lags, past-only rolling statistics, humidity, wind, calendar terms). The
target is the heatwave label `h` days ahead (`h = 1` by default, configurable).
Data is split chronologically into train, validation and test sets. Models:
climatology baseline, persistence baseline, Logistic Regression, Random Forest.
Metrics: precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix and
calibration. Accuracy alone is not used because heatwave days are rare.

**Warning logic.** Thresholds are chosen on validation data from the stated cost
of a missed event versus a false alarm. The ML probability and the EVT
return-period context are shown side by side and never merged into one
probability, because they answer different questions and use the same
underlying temperature evidence.

## Roadmap

### Part 1: Data and Extreme Value Theory
- [ ] Project configuration (`config/config.yaml`), logging, auto-created folders
- [ ] Download and inspect IMD and NASA POWER data (`src/data_download.py`)
- [ ] Clean and validate units, dates, missing values (`src/data_preprocessing.py`)
- [ ] Exploratory analysis: time series, anomalies, seasonal cycle, trends
- [ ] Block maxima, GEV fit, return levels with confidence intervals
- [ ] Diagnostics and stationarity check (`src/evt_analysis.py`)
- [ ] Unit tests for EVT functions

### Part 2: Labels and Machine Learning
- [ ] IMD-style proxy heatwave labels with class-balance report
- [ ] Leakage-safe features with configurable forecast horizon
- [ ] Temporal-leakage tests
- [ ] Climatology and persistence baselines
- [ ] Logistic Regression and Random Forest, chronological split
- [ ] Evaluation: precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix
- [ ] Probability calibration and calibration curve

### Part 3: Warning System, Dashboard and Documentation
- [ ] Cost-based warning thresholds chosen on validation data
- [ ] Risk categories with contributing factors and general safety guidance
- [ ] EVT context shown separately from the ML probability
- [ ] Streamlit dashboard: temperature history, anomalies, return levels,
      next-day probability, past events, model results and limitations
- [ ] Dashboard works on a small sample dataset
- [ ] Full test run, reproducibility check, deployment instructions
- [ ] Architecture diagram, data-flow diagram, final report, presentation and
      viva preparation

## Project structure

```
heatwave-early-warning/
├── README.md
├── requirements.txt
├── .env.example
├── config/config.yaml
├── data/{raw,processed}/
├── notebooks/01_exploratory_analysis.ipynb
├── src/
│   ├── data_download.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── heatwave_labels.py
│   ├── evt_analysis.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_warning.py
├── app/streamlit_app.py
├── models/
├── outputs/{figures,tables}/
└── tests/
```

## Installation

Requires Python 3.10 to 3.12.

```bash
git clone https://github.com/shannon-fernandes/heatwave-early-warning.git
cd heatwave-early-warning
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Planned run order. Each command is documented with its expected output once
the script is built and tested.

```bash
python -m src.data_download
python -m src.data_preprocessing
python -m src.evt_analysis
python -m src.heatwave_labels
python -m src.train_model
python -m src.evaluate_model
streamlit run app/streamlit_app.py
pytest
```

## Results

Results will be added here only after the code that produces them has been run.
Planned tables: GEV parameters with confidence intervals, return levels for 2,
5, 10 and 20 years, a model comparison on the test period, and the number of
heatwave days in each data split (very few events make scores unreliable).

## Limitations

- Gridded values are area averages and tend to understate local extremes, and
  coastal cells mix land and sea
- The proxy label is not an official IMD heatwave declaration
- Heatwaves in Chennai are rare, so class imbalance and small test samples
  limit how far the scores can be trusted
- Only observed data is used; operational systems use numerical weather forecasts
- GEV assumes stationarity, which is tested and not taken for granted
- Long-period return levels are uncertain with a limited number of years

## Future scope

Non-stationary GEV, peaks-over-threshold (GPD) comparison, additional cities,
numerical forecast inputs, spatial risk maps, ERA5-Land validation.

## Literature search terms

Search Google Scholar or Scopus, then read and cite real papers yourself:
"heatwave India IMD criteria", "generalized extreme value temperature return
level", "block maxima vs peaks over threshold", "heatwave prediction machine
learning", "probability calibration rare events", "Chennai extreme temperature
trend".

## Data attribution

Data comes from the India Meteorological Department, NASA POWER and, optionally,
the Copernicus Climate Change Service. Each provider's required attribution
wording is added here when its data is used in published results.

## Team

- Shannon Fernandes (GitHub: shannon-fernandes)
- Teammates are added here when they join the repository.

## License

No license has been chosen yet, so all rights are reserved by default. A license
will be added before the repository is made public. Data licenses are separate
from code licenses.