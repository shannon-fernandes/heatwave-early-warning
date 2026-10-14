import numpy as np
import pandas as pd
import pytest

from app import evt_panel


def _frame(years=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-01", periods=int(years * 365.25), freq="D")
    doy = idx.dayofyear.to_numpy()
    t = 33 + 4 * np.sin((doy - 80) / 365.25 * 2 * np.pi) + rng.gumbel(0, 1.0, len(idx))
    return pd.DataFrame({"date": idx, "max_temp_c": t})


def test_results_have_table_and_stationarity_info():
    r = evt_panel.compute_panel_results(_frame(), n_boot=30)
    assert list(r["ci"]["T"]) == [2, 5, 10, 20]
    assert (r["ci"]["lo95"] <= r["ci"]["hi95"]).all()
    assert 0 <= r["trend_gev"].p_value <= 1


def test_short_record_returns_message_not_exception():
    r = evt_panel.compute_panel_results(_frame(years=8), n_boot=10)
    assert "error" in r and "at least" in r["error"]


def test_missing_column_returns_message():
    r = evt_panel.compute_panel_results(_frame().drop(columns="max_temp_c"))
    assert "error" in r


def test_blank_temperatures_are_tolerated():
    f = _frame()
    f.loc[f.sample(500, random_state=1).index, "max_temp_c"] = np.nan
    assert "error" not in evt_panel.compute_panel_results(f, n_boot=10)


def test_panel_renders_in_streamlit():
    from streamlit.testing.v1 import AppTest

    def app():
        import numpy as np
        import pandas as pd

        from app.evt_panel import render_return_level_panel

        rng = np.random.default_rng(0)
        idx = pd.date_range("1960-01-01", periods=int(50 * 365.25), freq="D")
        doy = idx.dayofyear.to_numpy()
        t = 33 + 4 * np.sin((doy - 80) / 365.25 * 2 * np.pi) + rng.gumbel(0, 1.0, len(idx))
        render_return_level_panel(pd.DataFrame({"date": idx, "max_temp_c": t}))

    at = AppTest.from_function(app, default_timeout=120).run()
    assert not at.exception
    assert any("Extreme-value context" in s.value for s in at.subheader)
    assert any("Stationarity check" in m.value for m in at.markdown)


def test_figures_never_touch_pyplot_so_they_are_safe_in_streamlit_threads():
    """Regression: pyplot's macOS backend crashes off the main thread. Run in a fresh interpreter."""
    import subprocess
    import sys

    code = (
        "import sys, numpy as np, pandas as pd\n"
        "from src import evt_analysis as evt, evaluate_model as ev\n"
        "x = pd.Series(np.linspace(40, 44, 30), index=range(1990, 2020))\n"
        "fit = evt.fit_gev(x); ci = evt.bootstrap_gev_return_levels(x, n_boot=10)\n"
        "evt.return_level_figure(x, fit, ci); evt.diagnostics_figure(x, fit)\n"
        "y = np.r_[np.zeros(80), np.ones(20)]; p = np.linspace(0, 1, 100)\n"
        "ev.plot_calibration({'m': ev.calibration_table(y, p)}); ev.plot_pr_curves(y, {'m': p})\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was imported'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
