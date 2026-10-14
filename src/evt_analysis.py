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
