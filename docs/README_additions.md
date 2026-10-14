# Paste-ready README changes

## Replace the "Heatwave labels" paragraph (Methodology) with
The IMD-style rule (Tmax >= 37 C and departure >= 4.5 C on 2 consecutive days) produces almost no events at
Chennai: 6 positive days in 1981-2012 and 5 in 2013-2020. A classifier cannot be trained or evaluated on that.
The working target is therefore a **severe hot-day proxy**: Tmax >= 38 C and departure from normal >= 3.5 C on
2 consecutive days (84 positive days in training, 28 in validation). It is NOT an official IMD heatwave. The IMD-style
rule is still evaluated and reported (see `outputs/tables/label_counts.csv`). Departure from normal is Tmax minus the
mean Tmax for that day of year over 1951-2005, a short fixed reference period rather than an official 30-year normal.

## Add to "Usage" (replace the planned run order)
```bash
python -m src.data_download            # or copy the OpenCity CSV to data/raw/chennai_opencity_daily.csv
python -m src.data_preprocessing
python -m src.evt_analysis             # GEV return levels -> outputs/
python -m src.export_dashboard_csv     # dashboard/ML input -> data/processed + data/sample
python -m src.run_evaluation           # model vs baselines -> outputs/
python -m streamlit run app/streamlit_app.py
python -m pytest
```
(macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.)

## Add to "Methodology": evaluation protocol
Chronological split (80% train, 20% validation by date, 1981-2020). The validation period is cut in half by date:
the first half chooses each model's warning threshold (a missed event costs 10 false alarms, an assumption), the second
half is used only for scoring. Baselines: climatology (training-years rate for that day of year) and persistence
(yesterday's label). Predictors: calendar-day lags, past-only rolling Tmax, lagged departure from normal, and humidity and
wind when available. The lag settings were fixed in advance and not tuned on the scoring half.

## Results: paste from the generated files only
- EVT: `outputs/tables/gev_return_levels.csv`, `gev_parameters.csv` (primary fit 1951-2020; see earlier README_additions).
- Model: `outputs/tables/model_comparison_severe_hot_day_proxy.csv`, `calibration_*.csv`, `label_counts.csv`.
State plainly whatever the table shows, including that persistence is a strong baseline and that the
holdout has few events (a confidence interval for PR-AUC is in the run output).

## Limitations to add
- Few events: the scoring half holds about 17 positive days, so every score is uncertain.
- Probabilities from a class-balanced logistic regression are over-confident; calibration is reported, not corrected.
- The grid-to-station switch in 2021 is excluded from the ML years (1981-2020) and from the primary EVT fit.
- Humidity and wind come from NASA POWER (a model reanalysis); if that download is skipped the model is temperature-only.

## Project structure additions
src/validation.py, src/warning_logic.py, src/export_dashboard_csv.py, src/run_evaluation.py, src/pipeline_labels.py
(batch-pipeline labels, the original heatwave_labels.py), src/pipeline_features.py, app/evt_panel.py,
scripts/apply_dashboard_patch.py, tests/test_*.py
