"""Extra results for the paper. Prints everything and saves to outputs/."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from src import run_evaluation as R, evt_analysis as e
from src.config import load_config, city_slug, get_path
from src.heatwave_labels import assign_heatwave_proxy_labels

cfg = load_config()
proc, tabs, figs = get_path(cfg, "processed_dir"), get_path(cfg, "tables_dir"), get_path(cfg, "figures_dir")
frame = pd.read_csv(proc / f"{city_slug(cfg)}_dashboard.csv", parse_dates=["date"])
SEV = R.LABEL_SPECS["severe_hot_day_proxy"]

print("\n=== A. FIRST-DAY DETECTION (holdout) ===")
out = R.evaluate_label(frame, "severe_hot_day_proxy", SEV)
dates = out["result"].validation_dates[len(out["result"].validation_dates) // 2:]
lab = assign_heatwave_proxy_labels(frame, **SEV)["heatwave_proxy"].set_axis(pd.DatetimeIndex(frame["date"])).sort_index()
prev = lab.reindex(pd.date_range(lab.index.min(), lab.index.max())).shift(1).reindex(dates)
prev = prev.astype("Float64").fillna(0).astype(float).to_numpy()
y, p = out["y_hold"], out["probs_hold"]["logistic_regression"]
thr = out["thresholds"]["logistic_regression"]
first = (y == 1) & (prev == 0); cont = (y == 1) & (prev == 1); quiet = (y == 0) & (prev == 0)
warn = p >= thr
print(f"event days: {int(y.sum())}  first-day: {int(first.sum())}  continuing: {int(cont.sum())}")
print(f"model catches first-day events: {int((warn & first).sum())}/{int(first.sum())}  (persistence: 0 by construction)")
print(f"model catches continuing events: {int((warn & cont).sum())}/{int(cont.sum())}")
print(f"model false alarms on days after a non-event day: {int((warn & quiet).sum())}/{int(quiet.sum())}")

print("\n=== B. COST-RATIO SENSITIVITY ===")
rows = []
for k in (5, 10, 20, 50):
    R.COST_MISSED = float(k)
    o = R.evaluate_label(frame, "severe_hot_day_proxy", SEV)
    t = o["table"].set_index("model").loc["logistic_regression"]
    rows.append(dict(cost_ratio=k, threshold=t.threshold, precision=t.precision, recall=t.recall, f1=t.f1, tp=t.tp, fp=t.fp, fn=t.fn))
R.COST_MISSED = 10.0
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== C. LABEL-THRESHOLD SENSITIVITY ===")
rows = []
for tmax, dep in ((38, 3.5), (38, 3.0), (38, 4.0), (37, 3.5), (39, 3.5)):
    spec = dict(max_temperature_threshold_c=tmax, departure_threshold_c=dep, minimum_consecutive_days=2)
    try:
        o = R.evaluate_label(frame, f"{tmax}_{dep}", spec); tb = o["table"].set_index("model")
        rows.append(dict(tmax=tmax, dep=dep, n_pos=int(tb.loc["persistence", "n_positive"]),
                         lr_f1=tb.loc["logistic_regression", "f1"], lr_prauc=tb.loc["logistic_regression", "pr_auc"],
                         lr_roc=tb.loc["logistic_regression", "roc_auc"], pers_f1=tb.loc["persistence", "f1"], pers_prauc=tb.loc["persistence", "pr_auc"]))
    except Exception as ex:
        rows.append(dict(tmax=tmax, dep=dep, n_pos="failed: %s" % ex))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== D. TOP COEFFICIENTS (training-fit model, standardized features) ===")
m = out["result"].model
coef = pd.Series(m[-1].coef_[0], index=out["result"].feature_columns)
print(pd.concat([coef.sort_values(ascending=False).head(6), coef.sort_values().head(6)]).round(3).to_string())

print("\n=== E. GPD CROSS-CHECK (1951-2020, declustered, run_length=3) ===")
s = e.restrict_years(e.load_processed_tmax(str(proc / f"{city_slug(cfg)}_daily.csv")), end=2020)
for u in (38.0, 39.0, 40.0):
    try:
        g = e.fit_gpd(s, u)
        print(f"u={u}: peaks={g.n_peaks} xi={g.xi:.3f} sigma={g.sigma:.3f} RL10/50/100="
              + "/".join(f"{float(g.return_level(T)):.2f}" for T in (10, 50, 100)))
    except Exception as ex:
        print(f"u={u}: {ex}")

print("\n=== F. TREND-MODEL RETURN LEVELS (1951 vs 2020) and top maxima ===")
bm = e.block_maxima(s); tf = e.fit_gev_trend(bm)
for T in (2, 5, 10, 20, 50, 100):
    print(f"T={T}: 1951={float(e.trend_return_level(tf,1951,T)):.2f}  2020={float(e.trend_return_level(tf,2020,T)):.2f}")
print("top 5 seasonal maxima:\n", bm.sort_values(ascending=False).head(5).to_string())
sl, ic = stats.theilslopes(bm.to_numpy(), bm.index.to_numpy())[:2]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.scatter(bm.index, bm.values, s=14, c="k", label="seasonal maxima")
ax.plot(bm.index, ic + sl * bm.index.to_numpy(), label=f"Theil-Sen ({sl*10:.2f} °C/decade)")
ax.set(xlabel="year", ylabel="Tmax (°C)", title="Seasonal (Mar-Jun) maxima, 1951-2020"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(figs / "seasonal_maxima_trend.png", dpi=150)
print("saved", figs / "seasonal_maxima_trend.png")