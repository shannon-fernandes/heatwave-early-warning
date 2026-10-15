from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import update_live as ul


def _payload(days=40, missing_tail=3):
    idx = pd.date_range("2026-08-01", periods=days)
    def col(base):
        v = [base + i * 0.01 for i in range(days)]
        for k in range(missing_tail):
            v[-1 - k] = -999
        return {d.strftime("%Y%m%d"): x for d, x in zip(idx, v)}
    return {"properties": {"parameter": {"T2M_MAX": col(34.0), "T2M_MIN": col(26.0), "RH2M": col(70.0), "WS2M": col(3.0)}}}


def test_parse_drops_unpublished_tail_and_fill_values():
    df = ul.parse_power_payload(_payload(40, missing_tail=3))
    assert len(df) == 37 and df["max_temp_c"].notna().all()
    assert list(df.columns) == ["date", "max_temp_c", "min_temp_c", "humidity", "wind_speed"]


def test_parse_keeps_isolated_gaps_inside_the_series():
    p = _payload(20, missing_tail=0)
    p["properties"]["parameter"]["RH2M"]["20260805"] = -999
    df = ul.parse_power_payload(p)
    assert len(df) == 20 and df["humidity"].isna().sum() == 1


@pytest.mark.parametrize("bad", [{}, {"properties": {}}, None])
def test_unexpected_response_is_a_clear_error(bad):
    with pytest.raises(RuntimeError, match="Unexpected NASA POWER response"):
        ul.parse_power_payload(bad)


def test_no_temperature_at_all_is_a_clear_error():
    p = _payload(5, missing_tail=5)
    with pytest.raises(RuntimeError, match="no temperature"):
        ul.parse_power_payload(p)


class _Resp:
    def __init__(self, ok, payload=None, code=200):
        self.ok, self._p, self.status_code, self.text = ok, payload, code, "err"

    def json(self):
        return self._p


def test_fetch_backs_off_to_an_earlier_end_date(monkeypatch):
    seen = []

    def fake_get(url, params, timeout):
        seen.append(params["end"])
        return _Resp(False, code=422) if len(seen) < 3 else _Resp(True, _payload())

    monkeypatch.setattr(ul.requests, "get", fake_get)
    df = ul.fetch_power(date(2026, 8, 1), 13.08, 80.27, today=date(2026, 10, 15))
    assert seen == ["20261014", "20261011", "20261005"] and len(df) > 0


def test_fetch_gives_up_clearly(monkeypatch):
    monkeypatch.setattr(ul.requests, "get", lambda *a, **k: _Resp(False, code=500))
    with pytest.raises(RuntimeError, match="refused every end date"):
        ul.fetch_power(date(2026, 8, 1), 13.08, 80.27, today=date(2026, 10, 15))


def test_network_failure_is_a_clear_error(monkeypatch):
    def boom(*a, **k):
        raise ul.requests.ConnectionError("offline")
    monkeypatch.setattr(ul.requests, "get", boom)
    with pytest.raises(RuntimeError, match="Could not reach NASA POWER"):
        ul.fetch_power(date(2026, 8, 1), 13.08, 80.27, today=date(2026, 10, 15))


def test_departure_uses_reference_years_only_and_columns_match_dashboard():
    idx = pd.date_range("1990-01-01", "2026-06-30")
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"date": idx, "max_temp_c": 33 + rng.normal(0, 1, len(idx)), "min_temp_c": 25.0,
                      "humidity": 70.0, "wind_speed": 3.0})
    out = ul.with_departure(p, 2005)
    assert list(out.columns) == ul.COLUMNS
    ref = p[p.date.dt.year <= 2005].set_index("date")["max_temp_c"]
    clim = ref.groupby(ref.index.dayofyear).mean()
    r = out.iloc[-100]
    assert r.departure_from_normal_c == pytest.approx(r.max_temp_c - clim[pd.Timestamp(r.date).dayofyear])
    p2 = p.copy(); p2.loc[p2.date.dt.year >= 2010, "max_temp_c"] += 10        # later years must not move the normal
    assert ul.with_departure(p2, 2005).iloc[100].departure_from_normal_c == out.iloc[100].departure_from_normal_c


def test_recent_gap_counter():
    f = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "a": 1.0, "b": 2.0})
    f.loc[8, "b"] = np.nan
    assert ul.recent_gaps(f) == 1
