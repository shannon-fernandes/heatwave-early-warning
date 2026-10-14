"""Extreme value analysis of Chennai daily maximum temperature.

Person B, Part 1. Works on a pandas Series of daily Tmax with a DatetimeIndex,
so it does not depend on Person A's preprocessing (a minimal loader is
included for the raw OpenCity CSV).

Convention note: scipy's genextreme uses shape c = -xi.
  xi > 0 heavy tail, xi = 0 Gumbel, xi < 0 bounded upper tail.
All functions here report xi (the usual EVT sign), not scipy's c.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SUMMER_MONTHS = (3, 4, 5, 6)  # Mar-Jun: Chennai pre-monsoon heatwave season


# --------------------------------------------------------------------- loading
def load_raw_tmax(path: str) -> pd.Series:
    """Minimal loader for the OpenCity CSV (dd-mm-yyyy, '-----' = missing)."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    s = pd.to_numeric(df["Temp Max"], errors="coerce")
    s.index = idx
    s.name = "tmax"
    return s.sort_index()


def summer_only(s: pd.Series, months=SUMMER_MONTHS) -> pd.Series:
    return s[s.index.month.isin(months)]


# -------------------------------------------------------------- block maxima
def block_maxima(s: pd.Series, min_obs: int = 60, months=SUMMER_MONTHS) -> pd.Series:
    """Annual maximum of the summer season. Years with too few obs are dropped
    (e.g. a partial final year) so the maxima are comparable."""
    s = summer_only(s.dropna(), months)
    g = s.groupby(s.index.year)
    out = g.max()[g.count() >= min_obs]
    out.index.name = "year"
    return out


@dataclass
class GEVFit:
    xi: float
    loc: float
    scale: float
    n: int

    def return_level(self, T):
        return gev_return_level(self.xi, self.loc, self.scale, T)


def fit_gev(maxima) -> GEVFit:
    x = np.asarray(maxima, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        raise ValueError("Need at least 10 block maxima to fit a GEV.")
    c, loc, scale = stats.genextreme.fit(x)
    return GEVFit(xi=-c, loc=loc, scale=scale, n=len(x))


def gev_return_level(xi, loc, scale, T):
    """Level exceeded on average once every T blocks (years)."""
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1):
        raise ValueError("Return period must be > 1.")
    p = 1 - 1 / T
    return stats.genextreme.ppf(p, c=-xi, loc=loc, scale=scale)


def gev_exceedance_prob(fit: GEVFit, level: float) -> float:
    """P(annual summer max > level)."""
    return float(stats.genextreme.sf(level, c=-fit.xi, loc=fit.loc, scale=fit.scale))


# ------------------------------------------------------ peaks over threshold
def decluster_runs(s: pd.Series, threshold: float, run_length: int = 3) -> pd.Series:
    """Runs declustering: exceedances separated by fewer than `run_length`
    non-exceeding days belong to one cluster; keep only each cluster's max.
    Needed because hot days come in spells, which breaks GPD independence."""
    s = s.dropna()
    exc = s[s > threshold]
    if exc.empty:
        return exc
    # position of each exceedance in the full series
    pos = np.flatnonzero((s > threshold).to_numpy())
    cluster_id = np.concatenate([[0], np.cumsum(np.diff(pos) > run_length)])
    return exc.groupby(cluster_id).max().set_axis(
        exc.groupby(cluster_id).idxmax().to_numpy()
    )


def mean_residual_life(s: pd.Series, thresholds) -> pd.DataFrame:
    """Mean excess over a grid of thresholds (look for the linear region)."""
    s = s.dropna()
    rows = []
    for u in thresholds:
        e = s[s > u] - u
        if len(e) >= 10:
            se = e.std(ddof=1) / np.sqrt(len(e))
            rows.append((u, e.mean(), e.mean() - 1.96 * se, e.mean() + 1.96 * se, len(e)))
    return pd.DataFrame(rows, columns=["u", "mean_excess", "lo", "hi", "n_exc"])


def threshold_stability(s: pd.Series, thresholds, run_length: int = 3) -> pd.DataFrame:
    """GPD xi and scale-shape (modified scale) across thresholds, after declustering."""
    rows = []
    for u in thresholds:
        peaks = decluster_runs(s, u, run_length)
        if len(peaks) < 15:
            continue
        xi, _, sigma = stats.genpareto.fit(peaks.to_numpy() - u, floc=0)
        rows.append((u, xi, sigma - xi * u, len(peaks)))
    return pd.DataFrame(rows, columns=["u", "xi", "mod_scale", "n_peaks"])


@dataclass
class GPDFit:
    xi: float
    sigma: float
    threshold: float
    n_peaks: int
    rate_per_year: float  # cluster peaks per year, used to convert to return periods

    def return_level(self, T_years):
        return gpd_return_level(self, T_years)


def fit_gpd(s: pd.Series, threshold: float, run_length: int = 3, n_years: float | None = None) -> GPDFit:
    s = s.dropna()
    peaks = decluster_runs(s, threshold, run_length)
    if len(peaks) < 15:
        raise ValueError(f"Only {len(peaks)} cluster peaks above {threshold}; lower the threshold.")
    xi, _, sigma = stats.genpareto.fit(peaks.to_numpy() - threshold, floc=0)
    if n_years is None:
        n_years = len(s) / 365.25
    return GPDFit(xi, sigma, threshold, len(peaks), len(peaks) / n_years)


def gpd_return_level(fit: GPDFit, T_years):
    """x_T = u + sigma/xi * ((lambda*T)^xi - 1); Gumbel limit as xi -> 0."""
    T = np.asarray(T_years, dtype=float)
    m = fit.rate_per_year * T
    if np.any(m <= 1):
        raise ValueError("Return period too short for this threshold/rate.")
    if abs(fit.xi) < 1e-6:
        return fit.threshold + fit.sigma * np.log(m)
    return fit.threshold + fit.sigma / fit.xi * (m ** fit.xi - 1)


# ------------------------------------------------------------- uncertainty
def bootstrap_gev_return_levels(maxima, T=(10, 25, 50, 100), n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Parametric bootstrap CIs (resample from the fitted GEV, refit)."""
    rng = np.random.default_rng(seed)
    fit = fit_gev(maxima)
    T = np.asarray(T, dtype=float)
    draws = []
    for _ in range(n_boot):
        samp = stats.genextreme.rvs(-fit.xi, fit.loc, fit.scale, size=fit.n, random_state=rng)
        try:
            draws.append(fit_gev(samp).return_level(T))
        except Exception:
            continue
    d = np.array(draws)
    return pd.DataFrame(
        {
            "T": T,
            "estimate": fit.return_level(T),
            "lo95": np.percentile(d, 2.5, axis=0),
            "hi95": np.percentile(d, 97.5, axis=0),
        }
    )


# -------------------------------------------------------- diagnostics / trend
def gev_qq(maxima, fit: GEVFit) -> pd.DataFrame:
    """Empirical vs model quantiles (Weibull plotting positions)."""
    x = np.sort(np.asarray(maxima, dtype=float))
    p = np.arange(1, len(x) + 1) / (len(x) + 1)
    return pd.DataFrame(
        {"model": stats.genextreme.ppf(p, c=-fit.xi, loc=fit.loc, scale=fit.scale), "empirical": x}
    )


def gev_ks_pvalue(maxima, fit: GEVFit) -> float:
    return float(stats.kstest(maxima, "genextreme", args=(-fit.xi, fit.loc, fit.scale)).pvalue)


def trend_test(maxima: pd.Series) -> dict:
    """Mann-Kendall-style check via Kendall tau on year vs maximum, plus Theil-Sen slope (C per decade)."""
    m = maxima.dropna()
    tau, p = stats.kendalltau(m.index.to_numpy(), m.to_numpy())
    slope = stats.theilslopes(m.to_numpy(), m.index.to_numpy())[0]
    return {"tau": float(tau), "p_value": float(p), "slope_per_decade": float(slope * 10)}


def compare_periods(maxima: pd.Series, split_year: int = 1991) -> pd.DataFrame:
    """GEV return levels before vs after split_year (crude nonstationarity check)."""
    rows = []
    for name, part in [("early", maxima[maxima.index < split_year]), ("late", maxima[maxima.index >= split_year])]:
        f = fit_gev(part)
        rl = f.return_level([10, 50])
        rows.append((name, len(part), f.xi, f.loc, f.scale, rl[0], rl[1]))
    return pd.DataFrame(rows, columns=["period", "n", "xi", "loc", "scale", "RL10", "RL50"])


def empirical_exceedance_by_year(s: pd.Series, level: float) -> pd.Series:
    """Days per year above `level` (e.g. 40 C), a model-free summary."""
    s = s.dropna()
    return (s > level).groupby(s.index.year).sum()


# ------------------------------------------- non-stationary GEV (trend in loc)
@dataclass
class TrendGEVFit:
    xi: float
    loc0: float          # location at reference year
    loc_slope: float     # change in location per year
    scale: float
    ref_year: int
    lr_stat: float       # likelihood-ratio statistic vs stationary GEV
    p_value: float       # chi-square(1)

    @property
    def slope_per_decade(self) -> float:
        return self.loc_slope * 10


def fit_gev_trend(maxima: pd.Series) -> TrendGEVFit:
    """GEV with linear trend in the location parameter, compared to the
    stationary GEV by a likelihood-ratio test (df = 1). Answers: is a
    stationary GEV defensible for these maxima?"""
    from scipy.optimize import minimize

    m = maxima.dropna()
    if len(m) < 20:
        raise ValueError("Need at least 20 block maxima for a trend GEV.")
    yrs = m.index.to_numpy(dtype=float)
    ref = int(yrs.min())
    t = yrs - ref
    x = m.to_numpy(dtype=float)

    def nll(p):
        mu0, mu1, log_s, xi = p
        s = np.exp(log_s)
        v = stats.genextreme.logpdf(x, c=-xi, loc=mu0 + mu1 * t, scale=s)
        return 1e10 if not np.all(np.isfinite(v)) else -v.sum()

    st = fit_gev(m)
    start = [st.loc, 0.0, np.log(st.scale), st.xi]
    best = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    ll_trend = -best.fun
    ll_stat = stats.genextreme.logpdf(x, c=-st.xi, loc=st.loc, scale=st.scale).sum()
    lr = max(0.0, 2 * (ll_trend - ll_stat))
    mu0, mu1, log_s, xi = best.x
    return TrendGEVFit(xi, mu0, mu1, float(np.exp(log_s)), ref, float(lr), float(stats.chi2.sf(lr, 1)))


def trend_return_level(fit: TrendGEVFit, year: int, T) -> np.ndarray:
    """Return level under the trend model for a given year (conditional on that year's climate)."""
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1):
        raise ValueError("Return period must be > 1.")
    loc = fit.loc0 + fit.loc_slope * (year - fit.ref_year)
    return stats.genextreme.ppf(1 - 1 / T, c=-fit.xi, loc=loc, scale=fit.scale)


# ------------------------------------------------------------ loading (A's file)
def load_processed_tmax(path: str, column: str = "tmax") -> pd.Series:
    """Read Person A's processed daily file (data/processed/<city>_daily.csv)."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not in {path}. Columns: {list(df.columns)}")
    s = df[column].astype(float).sort_index()
    s.name = "tmax"
    return s


def restrict_years(s: pd.Series, start: int | None = None, end: int | None = None) -> pd.Series:
    if start is not None:
        s = s[s.index.year >= start]
    if end is not None:
        s = s[s.index.year <= end]
    return s


# ------------------------------------------------------------------- figures
def return_level_figure(maxima, fit: GEVFit, ci: pd.DataFrame, title="GEV return levels"):
    """Return-level plot with 95% bootstrap band and empirical points."""

    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    T = np.logspace(np.log10(1.5), np.log10(max(ci["T"].max(), 100)), 120)
    ax.semilogx(T, fit.return_level(T), label="GEV fit")
    ax.fill_between(ci["T"], ci["lo95"], ci["hi95"], alpha=0.2, label="95% CI")
    x = np.sort(np.asarray(maxima, dtype=float))[::-1]
    ax.scatter((len(x) + 1) / np.arange(1, len(x) + 1), x, s=14, c="k", label="annual maxima")
    ax.set_xlabel("return period (years)")
    ax.set_ylabel("Tmax (°C)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def diagnostics_figure(maxima, fit: GEVFit):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots(1, 2)
    q = gev_qq(maxima, fit)
    ax[0].scatter(q.model, q.empirical, s=14)
    lim = [q.min().min(), q.max().max()]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="model quantile", ylabel="empirical", title="GEV QQ plot")
    x = np.asarray(maxima, dtype=float)
    ax[1].hist(x, bins=15, density=True, alpha=0.5)
    xs = np.linspace(x.min() - 1, x.max() + 2, 200)
    ax[1].plot(xs, stats.genextreme.pdf(xs, -fit.xi, fit.loc, fit.scale))
    ax[1].set(title="Fitted density", xlabel="annual max Tmax (°C)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ pipeline
RETURN_PERIODS = (2, 5, 10, 20)       # the README's research question 1
EXTRA_PERIODS = (50, 100)             # shown as extrapolation only


def analyse_period(s: pd.Series, label: str, n_boot: int = 500) -> dict:
    bm = block_maxima(s)
    fit = fit_gev(bm)
    T = list(RETURN_PERIODS) + list(EXTRA_PERIODS)
    ci = bootstrap_gev_return_levels(bm, T=T, n_boot=n_boot)
    ci.insert(0, "period", label)
    trend = trend_test(bm)
    tfit = fit_gev_trend(bm)
    params = pd.DataFrame([{
        "period": label, "first_year": int(bm.index.min()), "last_year": int(bm.index.max()),
        "n_years": fit.n, "xi": fit.xi, "loc": fit.loc, "scale": fit.scale,
        "ks_p": gev_ks_pvalue(bm, fit), "kendall_tau": trend["tau"], "kendall_p": trend["p_value"],
        "theil_sen_per_decade": trend["slope_per_decade"],
        "trend_gev_loc_per_decade": tfit.slope_per_decade, "lrt_p_stationary_vs_trend": tfit.p_value,
        "RL10_trend_model_first_year": float(trend_return_level(tfit, int(bm.index.min()), 10)),
        "RL10_trend_model_last_year": float(trend_return_level(tfit, int(bm.index.max()), 10)),
    }])
    return {"maxima": bm, "fit": fit, "ci": ci, "params": params}


def run_analysis(cfg: dict, n_boot: int = 500) -> dict:
    """Primary fit on years up to analysis.homogeneous_end_year (grid-based IMD
    record, no station splice); sensitivity fit on the full record."""
    from src.config import city_slug, get_path

    path = get_path(cfg, "processed_dir") / f"{city_slug(cfg)}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_preprocessing` first.")
    s = load_processed_tmax(str(path))
    hom_end = cfg["analysis"]["homogeneous_end_year"]
    out = {
        "primary": analyse_period(restrict_years(s, end=hom_end), f"1951-{hom_end} (primary)", n_boot),
        "full": analyse_period(s, f"1951-{int(s.index.year.max())} (sensitivity)", n_boot),
    }
    tables, figures = get_path(cfg, "tables_dir"), get_path(cfg, "figures_dir")
    pd.concat([v["params"] for v in out.values()]).to_csv(tables / "gev_parameters.csv", index=False)
    pd.concat([v["ci"] for v in out.values()]).to_csv(tables / "gev_return_levels.csv", index=False)
    p = out["primary"]
    return_level_figure(p["maxima"], p["fit"], p["ci"], f"GEV return levels, {p['params'].period[0]}").savefig(
        figures / "gev_return_levels.png", dpi=150)
    diagnostics_figure(p["maxima"], p["fit"]).savefig(figures / "gev_diagnostics.png", dpi=150)
    return out


def main() -> None:
    import logging
    from src.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    out = run_analysis(load_config())
    for k, v in out.items():
        log.info("%s\n%s\n%s", k, v["params"].round(3).to_string(index=False), v["ci"].round(2).to_string(index=False))
    log.info("Wrote outputs/tables/gev_parameters.csv, gev_return_levels.csv and outputs/figures/gev_*.png")


if __name__ == "__main__":
    main()
