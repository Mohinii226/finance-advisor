"""
Rolling-origin backtest.

For each origin t the model sees ONLY columns 0..t and must predict column t+1. Metrics are pooled over all
(series, origin) pairs, restricted to the series every model was run on, so models are compared like for like.

  MAE   mean absolute error in rupees
  WAPE  sum|error| / sum|actual|  (scale-free, not distorted by tiny series)
  sMAPE mean 2|a-f|/(|a|+|f|)     (pairs where both are zero are skipped)
  MASE  |error| / in-sample one-step naive error of that series (1.0 = as good as last-month; <1 better)
  win   share of (series, origin) pairs where the model beats naive_last
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import Panel
from .models import EPS, Forecaster


def backtest(models: list[Forecaster], panel: Panel, n_origins: int = 12, max_series: dict[str, int] | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Returns long results: model, origin, series, actual, pred, mase_scale.  max_series: {model_name: cap} for slow models."""
    Y, months, meta = panel.Y, panel.months, panel.meta
    n, T = Y.shape
    origins = list(range(T - 1 - n_origins, T - 1))               # origin t predicts column t+1
    rows, timings = [], {m.name: 0.0 for m in models}
    for t in origins:
        for m in models:
            cap = (max_series or {}).get(m.name)
            idx = np.arange(n if cap is None else min(cap, n))
            t0 = time.time()
            pred = m.forecast(Y[idx, : t + 1], months[: t + 1], meta.iloc[idx].reset_index(drop=True))
            timings[m.name] += time.time() - t0
            hist = Y[idx, : t + 1]
            scale = np.abs(np.diff(hist, axis=1)).mean(axis=1) + EPS
            rows.append(pd.DataFrame({"model": m.name, "origin": t, "series": idx, "actual": Y[idx, t + 1],
                                      "pred": np.asarray(pred, float), "mase_scale": scale}))
        if verbose:
            print(f"  origin {t + 1}/{T - 1} done", flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.attrs["seconds"] = timings
    return res


def _common(res: pd.DataFrame) -> pd.DataFrame:
    common = set.intersection(*[set(g["series"]) for _, g in res.groupby("model")])
    return res[res["series"].isin(common)]


def summarise(res: pd.DataFrame, baseline: str = "naive_last") -> pd.DataFrame:
    r = _common(res).copy()
    r["abs_err"] = (r["pred"] - r["actual"]).abs()
    r["smape_term"] = np.where((r["actual"] + r["pred"]) > 0, 2 * r["abs_err"] / (r["actual"].abs() + r["pred"].abs()).replace(0, np.nan), np.nan)
    r["mase_term"] = r["abs_err"] / r["mase_scale"]
    base = r[r["model"] == baseline].set_index(["series", "origin"])["abs_err"]
    r["beats_base"] = r["abs_err"].to_numpy() < base.reindex(pd.MultiIndex.from_frame(r[["series", "origin"]])).to_numpy()
    g = r.groupby("model")
    out = pd.DataFrame({
        "MAE": g["abs_err"].mean(), "WAPE": g["abs_err"].sum() / g["actual"].apply(lambda s: s.abs().sum()),
        "sMAPE": g["smape_term"].mean(), "MASE": g["mase_term"].mean(), "win_vs_naive": g["beats_base"].mean(),
        "series": g["series"].nunique(), "origins": g["origin"].nunique(),
    })
    out["seconds"] = pd.Series(res.attrs.get("seconds", {}))
    return out.sort_values("WAPE")


def paired_bootstrap(res: pd.DataFrame, model: str, baseline: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI for WAPE(model) - WAPE(baseline), resampling SERIES (customers) with replacement."""
    r = _common(res)
    r = r.assign(e=(r["pred"] - r["actual"]).abs(), a=r["actual"].abs())
    per = r.groupby(["model", "series"]).agg(e=("e", "sum"), a=("a", "sum")).reset_index()
    m, b = per[per.model == model].set_index("series"), per[per.model == baseline].set_index("series")
    idx = m.index.intersection(b.index).to_numpy()
    em, eb, a = m.loc[idx, "e"].to_numpy(), b.loc[idx, "e"].to_numpy(), m.loc[idx, "a"].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(idx), len(idx))
        diffs.append((em[s].sum() - eb[s].sum()) / a[s].sum())
    point = (em.sum() - eb.sum()) / a.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(model=model, baseline=baseline, wape_diff=point, ci_low=lo, ci_high=hi, significant=bool(hi < 0 or lo > 0))


def interval_calibration(res: pd.DataFrame, model: str, level: float = 0.8) -> dict:
    """Relative-error quantiles from the FIRST half of origins define an interval; coverage is measured on the SECOND half."""
    r = _common(res)
    r = r[r["model"] == model].copy()
    r["rel"] = (r["actual"] - r["pred"]) / np.maximum(r["pred"], EPS)
    cut = np.sort(r["origin"].unique())[len(r["origin"].unique()) // 2]
    fit, test = r[r["origin"] < cut], r[r["origin"] >= cut]
    lo, hi = np.quantile(fit["rel"], [(1 - level) / 2, 1 - (1 - level) / 2])
    covered = ((test["actual"] >= test["pred"] * (1 + lo)) & (test["actual"] <= test["pred"] * (1 + hi))).mean()
    return dict(model=model, target=level, rel_low=float(lo), rel_high=float(hi), coverage=float(covered), n_test=len(test))
