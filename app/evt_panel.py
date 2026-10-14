"""EVT context panel for the Streamlit dashboard (Person B, Part 3).

Kept in its own module so it does not touch Person C's app code. It works on the
dashboard's uploaded frame (columns `date`, `max_temp_c`) and is shown
SEPARATELY from the ML probability: return levels answer "how rare is this
temperature", the classifier answers "will tomorrow be a heatwave day".
"""
from __future__ import annotations

import pandas as pd

from src import evt_analysis as evt

DATE_COLUMN = "date"
TMAX_COLUMN = "max_temp_c"
MIN_YEARS = 20          # fewer block maxima than this gives return levels too uncertain to show
PERIODS = (2, 5, 10, 20)


def compute_panel_results(dataset: pd.DataFrame, n_boot: int = 300) -> dict:
    """Pure function (no Streamlit): returns a dict, or {'error': message} if the data cannot support EVT."""
    for col in (DATE_COLUMN, TMAX_COLUMN):
        if col not in dataset.columns:
            return {"error": f"Column {col!r} is needed for the return-level analysis."}
    s = pd.Series(
        pd.to_numeric(dataset[TMAX_COLUMN], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(dataset[DATE_COLUMN])), name="tmax",
    ).sort_index()
    bm = evt.block_maxima(s)
    if len(bm) < MIN_YEARS:
        return {"error": f"Only {len(bm)} complete March to June season(s) in the uploaded file; "
                         f"at least {MIN_YEARS} are needed for a meaningful GEV fit."}
    fit = evt.fit_gev(bm)
    ci = evt.bootstrap_gev_return_levels(bm, T=list(PERIODS), n_boot=n_boot)
    out = {"maxima": bm, "fit": fit, "ci": ci, "trend": evt.trend_test(bm),
           "ks_p": evt.gev_ks_pvalue(bm, fit)}
    try:
        t = evt.fit_gev_trend(bm)
        out["trend_gev"] = t
        out["trend_rl10_now"] = float(evt.trend_return_level(t, int(bm.index.max()), 10))
    except ValueError:
        pass
    return out


def render_return_level_panel(dataset: pd.DataFrame) -> None:
    import streamlit as st

    st.subheader("Extreme-value context (separate from the ML probability)")
    st.caption("Return levels describe how rare a temperature is in the uploaded record. They are not "
               "a forecast and are not combined with the classifier's probability.")

    @st.cache_data(show_spinner="Fitting GEV and bootstrapping confidence intervals…")
    def _cached(frame: pd.DataFrame) -> dict:
        return compute_panel_results(frame)

    res = _cached(dataset[[DATE_COLUMN, TMAX_COLUMN]])
    if "error" in res:
        st.info(res["error"])
        return

    fit, ci, bm = res["fit"], res["ci"], res["maxima"]
    table = ci.rename(columns={"T": "Return period (years)", "estimate": "Return level (°C)",
                               "lo95": "95% CI low", "hi95": "95% CI high"}).round(2)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(table, hide_index=True, width="stretch")
    c1.caption(f"GEV fitted to {fit.n} annual March–June maxima ({bm.index.min()}–{bm.index.max()}). "
               f"Shape ξ = {fit.xi:.2f}; KS goodness-of-fit p = {res['ks_p']:.2f}.")
    fig = evt.return_level_figure(bm, fit, ci, "Annual maximum Tmax: return levels")
    c2.pyplot(fig)

    tr = res["trend"]
    if "trend_gev" in res:
        t = res["trend_gev"]
        verdict = ("evidence against a stationary model, so the table above may understate "
                   "present-day extremes" if t.p_value < 0.05 else "no clear evidence against stationarity")
        st.markdown(
            f"**Stationarity check.** Annual maxima change by about {tr['slope_per_decade']:.2f} °C per decade "
            f"(Theil–Sen; Kendall p = {tr['p_value']:.3f}). A GEV with a trending location improves on the "
            f"stationary fit with likelihood-ratio p = {t.p_value:.4f}: {verdict}. Under the trend model the "
            f"10-year level for {bm.index.max()} is about {res['trend_rl10_now']:.1f} °C.")
    st.caption("Return levels with long periods are extrapolations and carry wide uncertainty. "
               "If the upload mixes data sources (for example a grid-to-station switch), trends can be artefacts.")
