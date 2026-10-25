import numpy as np
import pandas as pd

from api.handlers import Backend
from core.service import AdvisorService, ToolError
from ml.forecasting.backtest import backtest, interval_calibration, paired_bootstrap, summarise
from ml.forecasting.data import Panel, monthly_total_panel
from ml.forecasting.models import (ARIMAModel, GlobalGBM, LSTMGlobal, MedianK, MovingAverage, NaiveLast, ProphetModel,
                                   SeasonalNaive, SimpleExpSmoothing, default_models)
from ml.forecasting.run_forecast import TABLE_COLUMNS, forecast_next_month
from tests.chat_fixture import chat_env
from tests.pg_mimic import pg_like


def _meta(n):
    return pd.DataFrame({"customer_id": [f"C{i}" for i in range(n)], "category": "TOTAL", "monthly_income": 50000.0, "income_type_code": 0})


def _synthetic(n=40, T=36, seasonal=True, seed=0, noise=0.08):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=T, freq="MS")
    level = rng.uniform(15000, 60000, n)[:, None]
    bump = np.where(np.isin(months.month, [11, 12]), 0.6, 0.0)[None, :] if seasonal else 0.0
    Y = level * (1 + bump) * rng.lognormal(0, noise, (n, T))
    return Panel(Y, months, _meta(n))


# ------------------------------------------------------------------ baselines and metrics
def test_baselines_on_toy_series():
    Y = np.array([[10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130.]])
    m, meta = pd.date_range("2024-01-01", periods=13, freq="MS"), _meta(1)
    assert NaiveLast().forecast(Y, m, meta)[0] == 130
    assert MovingAverage(3).forecast(Y, m, meta)[0] == 120
    assert MedianK(6).forecast(Y, m, meta)[0] == np.median(Y[0, -6:])
    assert SeasonalNaive().forecast(Y, m, meta)[0] == 20          # 12 months before the target month
    const = np.full((1, 24), 500.0)
    assert abs(SimpleExpSmoothing().forecast(const, pd.date_range("2024-01-01", periods=24, freq="MS"), meta)[0] - 500) < 1e-6


def test_metrics_exact_on_a_tiny_example():
    res = pd.DataFrame({"model": ["naive_last"] * 2 + ["m"] * 2, "origin": [1, 1, 1, 1], "series": [0, 1, 0, 1],
                        "actual": [100., 200., 100., 200.], "pred": [80., 260., 90., 210.], "mase_scale": [10., 10., 10., 10.]})
    t = summarise(res)
    assert abs(t.loc["naive_last", "MAE"] - 40) < 1e-9 and abs(t.loc["m", "MAE"] - 10) < 1e-9
    assert abs(t.loc["m", "WAPE"] - 20 / 300) < 1e-9 and t.loc["m", "win_vs_naive"] == 1.0 and abs(t.loc["m", "MASE"] - 1.0) < 1e-9


def test_paired_bootstrap_and_interval_calibration():
    rng = np.random.default_rng(1)
    n = 200
    act = rng.uniform(1000, 5000, n)
    base = act * (1 + rng.normal(0, 0.3, n))
    good = act * (1 + rng.normal(0, 0.1, n))
    mk = lambda name, p: pd.DataFrame({"model": name, "origin": 5, "series": range(n), "actual": act, "pred": p, "mase_scale": 1.0})
    res = pd.concat([mk("base", base), mk("good", good), mk("same", base)])
    assert paired_bootstrap(res, "good", "base")["significant"] and paired_bootstrap(res, "good", "base")["ci_high"] < 0
    same = paired_bootstrap(res, "same", "base")
    assert abs(same["wape_diff"]) < 1e-12 and not same["significant"]
    cal_rows = []
    for origin in range(10):
        a = 3000 * np.ones(400)
        p = a / (1 + rng.normal(0, 0.2, 400))
        cal_rows.append(pd.DataFrame({"model": "m", "origin": origin, "series": range(400), "actual": a, "pred": p, "mase_scale": 1.0}))
    c = interval_calibration(pd.concat(cal_rows), "m", 0.8)
    assert abs(c["coverage"] - 0.8) < 0.05, c


# ------------------------------------------------------------------ leakage and learning
def test_no_model_can_see_the_future():
    """Changing columns AFTER an origin must not change that origin's predictions (checked for every runnable model)."""
    P = _synthetic(n=30, T=30, seed=2)
    Y2 = P.Y.copy()
    Y2[:, -2:] *= 7.0                                   # corrupt the last two months
    P2 = Panel(Y2, P.months, P.meta)
    models = [NaiveLast(), MovingAverage(6), MedianK(6), SeasonalNaive(), SimpleExpSmoothing(), GlobalGBM("absolute_error"), GlobalGBM("squared_error")]
    r1, r2 = backtest(models, P, 6, verbose=False), backtest(models, P2, 6, verbose=False)
    T = P.Y.shape[1]
    safe = lambda r: r[r["origin"] <= T - 3].sort_values(["model", "origin", "series"])["pred"].to_numpy()
    assert len(safe(r1)) > 0 and np.allclose(safe(r1), safe(r2)), "a model's prediction depended on future data"


def test_gbm_learns_seasonality_that_trailing_averages_miss():
    P = _synthetic(n=60, T=36, seasonal=True, seed=3)
    res = backtest([MovingAverage(6), GlobalGBM("absolute_error")], P, 6, verbose=False)
    t = summarise(res, baseline="mean6")
    assert t.loc["gbm_l1", "WAPE"] < 0.8 * t.loc["mean6", "WAPE"], t


def test_gbm_outputs_are_finite_and_positive():
    P = _synthetic(n=25, T=24, seed=4)
    for loss in ("absolute_error", "squared_error"):
        out = GlobalGBM(loss).forecast(P.Y, P.months, P.meta)
        assert out.shape == (25,) and np.isfinite(out).all() and (out > 0).all()
    short = GlobalGBM().forecast(P.Y[:, :10], P.months[:10], P.meta)          # too little history -> graceful fallback
    assert np.isfinite(short).all()


# ------------------------------------------------------------------ data + batch job
def test_panel_matches_independent_pandas_total():
    e = chat_env()
    P = monthly_total_panel(e["txns"], e["D"]["d"]["customers"])
    t = e["txns"]
    cid = P.meta["customer_id"].iloc[3]
    d = t[(t.customer_id == cid) & (t.type == "debit")]
    d = d[~d.category.fillna("Uncategorized").isin(["Investments", "Self Transfer"])]
    m = pd.Timestamp("2025-06-01")
    exp = d[(pd.to_datetime(d.txn_ts) >= m) & (pd.to_datetime(d.txn_ts) < m + pd.DateOffset(months=1))].amount.sum()
    assert abs(P.Y[3, list(P.months).index(m)] - exp) < 0.05 and P.Y.shape[1] == 36


def _forecasts():
    e = chat_env()
    return forecast_next_month(monthly_total_panel(e["txns"], e["D"]["d"]["customers"]), n_origins=6)


def test_forecast_rows_match_table_and_are_coherent():
    f = _forecasts()
    assert list(f.columns) == TABLE_COLUMNS
    g = f[f.model_name == "gbm_l1"]
    assert len(g) == f.customer_id.nunique() and (g.lower_bound <= g.predicted_amount).all() and (g.predicted_amount <= g.upper_bound).all()
    assert (g.lower_bound >= 0).all() and set(f.target_month) == {pd.Timestamp("2026-01-01").date()}
    assert f[f.model_name == "median6"].lower_bound.isna().all()


def _with_forecast_table():
    e = chat_env()
    f = _forecasts()
    f["target_month"] = f["target_month"].astype(str)
    f.to_sql("forecasts", e["con"], index=False, if_exists="replace")
    return e


def test_service_forecast_and_chat_end_to_end():
    e = _with_forecast_table()
    s = AdvisorService(e["run"], e["cid"])
    r = s.get_spending_forecast()
    assert r["forecast_month"] == "January 2026" and r["expected_spend"] > 0 and f"{r['expected_spend']:,.0f}" in r["text"]
    assert r["likely_range_low"] <= r["expected_spend"] <= r["likely_range_high"]
    ans = Backend(e["run"]).chat(e["cid"], "How much will I spend next month?")
    assert ans["tool_calls"][0]["name"] == "get_spending_forecast" and ans["grounding"]["ok"] and "January 2026" in ans["answer"]
    pg = AdvisorService(pg_like(e["run"]), e["cid"]).get_spending_forecast()
    assert pg == r, "Postgres-style types must give the same forecast answer"


def test_service_forecast_reports_missing_job_cleanly():
    e = chat_env()

    def broken(name, **kw):
        if name == "forecast_for_customer":
            raise RuntimeError("relation forecasts does not exist")
        return e["run"](name, **kw)
    try:
        AdvisorService(broken, e["cid"]).get_spending_forecast(); assert False
    except ToolError as err:
        assert "not available yet" in str(err)


# ------------------------------------------------------------------ optional models: run only where their libraries are installed
def _skip(reason):
    import pytest
    pytest.skip(reason)


def test_arima_adapter_if_installed():
    if not ARIMAModel.available():
        _skip("statsmodels not installed (pip install statsmodels)")
    P = _synthetic(n=3, T=30, seasonal=False)
    out = ARIMAModel().forecast(P.Y, P.months, P.meta)
    assert out.shape == (3,) and np.isfinite(out).all() and (out >= 0).all()


def test_prophet_adapter_if_installed():
    if not ProphetModel.available():
        _skip("prophet not installed (pip install prophet)")
    P = _synthetic(n=2, T=30)
    out = ProphetModel().forecast(P.Y, P.months, P.meta)
    assert out.shape == (2,) and np.isfinite(out).all()


def test_lstm_adapter_if_installed():
    if not LSTMGlobal.available():
        _skip("torch not installed (pip install torch)")
    P = _synthetic(n=20, T=30)
    a = LSTMGlobal(epochs=3).forecast(P.Y, P.months, P.meta)
    b = LSTMGlobal(epochs=3).forecast(P.Y, P.months, P.meta)
    assert a.shape == (20,) and np.isfinite(a).all() and np.allclose(a, b), "seeded training should be repeatable"


def test_default_models_only_include_available_optional_ones():
    names = [m.name for m in default_models(True)]
    assert {"naive_last", "mean6", "median6", "ses", "gbm_l1", "gbm_l2"} <= set(names)
    assert ("arima" in names) == ARIMAModel.available() and ("prophet" in names) == ProphetModel.available()
