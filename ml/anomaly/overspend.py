"""
Overspending detector.

Input : a dense customer x category x month grid  (columns: customer_id, category, month, spend[, txn_count])
        e.g. `SELECT * FROM v_category_month_grid`
Output: ranked alerts with templated, explainable messages.

Why this is better than the SQL benchmark rule (v_overspend_alerts)
  1. Robust reference  - median / trimmed mean so one past spike does not poison the baseline.
  2. Peer adjustment   - expected spend is scaled by how *similar customers' spend moved the same month*,
                         absorbing festivals, seasonality and inflation without leaking future data.
  3. Persistence / level-shift - creeping overspend must be sustained (m of last n months, or the average of the
                         last L months must be abnormal); one-off spikes need a much higher bar.
  4. Reference gap     - the reference window skips the most recent months so a creeping category is not
                         absorbed into its own baseline.
  5. Frequency check   - purchase count must also rise (a single big ticket is a "large purchase", not a habit).
  6. Alert budget      - at most k alerts per customer-month, ranked by rupee impact (no alert fatigue).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 6                     # months in the reference window
    gap: int = 0                        # most-recent months skipped by the reference window
    center: str = "trimmed_mean"        # mean | trimmed_mean | median
    scale: str = "std_trim"             # std | std_trim | mad
    scale_floor_rel: float = 0.30       # scale >= this x center  (stops tiny scales exploding z)
    peer_adjust: bool = True
    peer_min_center: float = 100.0      # series with a smaller reference are ignored when computing peer ratios
    pct_thr: float = 0.40               # rule A/C: needs >= +40% vs expected
    z_thr: float = 2.0
    materiality: float = 500.0          # and >= Rs 500 above expected
    persist_n: int = 3                  # rule A: elevated in persist_m of the last persist_n months
    persist_m: int = 2
    persist_pct: float = 0.25
    spike_z: float = 6.0                # rule C: a single month alerts alone if z >= spike_z (np.inf disables)
    max_ref_cv: float = np.inf          # skip categories whose reference window is too erratic (std/center)
    min_count_ratio: float = 0.0        # frequency check: txn count >= this x reference count (needs txn_count column)
    use_persistence: bool = True        # rule A on/off
    shift_len: int = 0                  # rule B (level shift): mean of last L months vs reference; 0 = off.
    shift_z: float = 3.0                #   needs gap >= L-1 so the reference excludes the months being tested
    shift_pct: float = 0.30
    max_alerts: int | None = 2          # alert budget per customer-month (None = unlimited)
    exclude_categories: tuple = ("Uncategorized",)


NAIVE = DetectorConfig(center="mean", scale="std", scale_floor_rel=0.0, peer_adjust=False, pct_thr=0.25, z_thr=2.0,
                       materiality=500.0, persist_m=1, persist_n=1, persist_pct=0.0, spike_z=np.inf, max_alerts=None)
"""Re-implementation of the SQL benchmark rule, so both detectors share one evaluation path."""

# Operating points chosen on a DEV world (seed 11) and reported on an unseen TEST world (seed 22).
LOW_NOISE = DetectorConfig(window=6, gap=2, center="median", scale="std", scale_floor_rel=0.3, peer_adjust=True,
                           pct_thr=0.25, z_thr=4.0, materiality=1000.0, use_persistence=True, persist_n=3, persist_m=3,
                           persist_pct=0.25, shift_len=0, spike_z=np.inf, min_count_ratio=1.5, max_alerts=3)
"""Few, confident alerts: elevated 3 of the last 3 months, peer-adjusted, purchase frequency must also rise."""

BALANCED = DetectorConfig(window=6, gap=3, center="median", scale="std", scale_floor_rel=0.5, peer_adjust=False,
                          use_persistence=False, shift_len=4, shift_z=4.0, shift_pct=0.2, spike_z=7.0,
                          materiality=2000.0, min_count_ratio=1.2, max_alerts=1)
"""Level-shift test over the last 4 months (reference skips them) + high bar for one-off spikes; 1 alert/customer-month."""

PRESETS = {"naive": NAIVE, "balanced": BALANCED, "low_noise": LOW_NOISE}


# ------------------------------------------------------------------ panel helpers
def build_panel(grid: pd.DataFrame):
    wide = grid.pivot_table(index=["customer_id", "category"], columns="month", values="spend",
                            aggfunc="sum", fill_value=0.0)
    keys = wide.index.to_frame(index=False)
    months = pd.to_datetime(wide.columns)
    return keys, months, wide.to_numpy(dtype=float)


def _reference(S: np.ndarray, window: int, gap: int, center: str, scale: str):
    """Reference stats for month t from months [t-gap-window, t-gap-1]."""
    n, T = S.shape
    C = np.full((n, T), np.nan)
    SC = np.full((n, T), np.nan)
    M = np.full((n, T), np.nan)           # plain mean (used in user-facing messages)
    first = gap + window
    if T <= first:
        return C, SC, M
    win = sliding_window_view(S, window, axis=1)[:, : T - first, :]       # [n, T-first, window]
    srt = np.sort(win, axis=2)
    trimmed = srt[:, :, :-1] if window > 2 else srt
    med = np.median(win, axis=2)
    cen = {"mean": win.mean(2), "trimmed_mean": trimmed.mean(2), "median": med}[center]
    sc = {"std": win.std(2, ddof=1), "std_trim": trimmed.std(2, ddof=1),
          "mad": 1.4826 * np.median(np.abs(win - med[..., None]), axis=2)}[scale]
    C[:, first:], SC[:, first:], M[:, first:] = cen, sc, win.mean(2)
    return C, SC, M


def _peer_ratio(S, C, keys: pd.DataFrame, min_center: float) -> np.ndarray:
    """Median spend/reference ratio across customers, per category and month (1.0 = nothing unusual)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(C >= min_center, S / C, np.nan)
    pop = np.ones_like(S)
    for _, idx in keys.groupby("category").indices.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(ratio[idx], axis=0)
        pop[idx] = np.where(np.isnan(med), 1.0, med)
    return np.clip(pop, 0.5, 3.0)


def _rolling_mean(X: np.ndarray, L: int) -> np.ndarray:
    """Mean over the last L months including t (NaN for t < L-1)."""
    cs = np.cumsum(np.pad(X, ((0, 0), (1, 0))), axis=1)
    out = np.full_like(X, np.nan)
    out[:, L - 1:] = (cs[:, L:] - cs[:, : X.shape[1] - L + 1]) / L
    return out


# ------------------------------------------------------------------ core
def score_cells(grid: pd.DataFrame, cfg: DetectorConfig = None) -> pd.DataFrame:
    """Score every (customer, category, month) cell that has a full reference window."""
    cfg = BALANCED if cfg is None else cfg
    keys, months, S = build_panel(grid)
    T = S.shape[1]

    use_cnt = cfg.min_count_ratio > 0 and "txn_count" in grid.columns
    if use_cnt:
        _, _, NC = build_panel(grid.assign(spend=grid["txn_count"]))
        CC, _, _ = _reference(NC, cfg.window, cfg.gap, "mean", "std")

    C, SC, M = _reference(S, cfg.window, cfg.gap, cfg.center, cfg.scale)
    peer = _peer_ratio(S, C, keys, cfg.peer_min_center) if cfg.peer_adjust else np.ones_like(S)

    expected = C * peer
    scale = np.maximum(SC, cfg.scale_floor_rel * C) * peer
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(scale > 0, scale, np.nan)
        z = (S - expected) / scale
        adj_ratio = np.where(expected > 0, S / expected, np.nan)
        predictable = ~((SC / C) > cfg.max_ref_cv)                                  # NaN-safe
    excess = S - expected
    valid = ~np.isnan(C) & (expected > 0) & ~np.isnan(z)

    # frequency check (driver): purchase count must rise too
    count_ratio = np.full_like(S, np.nan)
    count_ok = np.ones_like(S, dtype=bool)
    if use_cnt:
        with np.errstate(divide="ignore", invalid="ignore"):
            count_ratio = np.where(CC > 0, NC / CC, np.nan)
        count_ok = count_ratio >= cfg.min_count_ratio

    # persistence: elevated in m of the last n months
    elevated = valid & (adj_ratio >= 1 + cfg.persist_pct)
    cs = np.cumsum(np.pad(elevated.astype(int), ((0, 0), (1, 0))), axis=1)
    lo = np.maximum(np.arange(1, T + 1) - cfg.persist_n, 0)
    elev_count = cs[:, 1:] - cs[:, lo]
    persistent = elevated & (elev_count >= cfg.persist_m)

    base = valid & predictable & count_ok & (adj_ratio >= 1 + cfg.pct_thr) & (excess >= cfg.materiality)
    flag = base & (z >= cfg.spike_z)                                                # rule C: one-off spike
    if cfg.use_persistence:                                                         # rule A
        flag |= base & persistent & (z >= cfg.z_thr)

    z_shift = np.full_like(S, np.nan)
    ratio_shift = np.full_like(S, np.nan)
    L = cfg.shift_len
    if L > 1:                                                                       # rule B: level shift
        S3, P3 = _rolling_mean(S, L), _rolling_mean(peer, L)
        E3 = C * P3
        sc3 = np.maximum(SC, cfg.scale_floor_rel * C) * P3 / np.sqrt(L)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_shift = np.where(sc3 > 0, (S3 - E3) / sc3, np.nan)
            ratio_shift = np.where(E3 > 0, S3 / E3, np.nan)
            cnt3_ok = (np.where(CC > 0, _rolling_mean(NC, L) / CC, np.nan) >= cfg.min_count_ratio) if use_cnt \
                else np.ones_like(S, dtype=bool)
        flag |= (valid & predictable & cnt3_ok & (z_shift >= cfg.shift_z) & (ratio_shift >= 1 + cfg.shift_pct)
                 & ((S3 - E3) >= cfg.materiality) & (S >= expected))

    if cfg.exclude_categories:
        flag &= ~keys["category"].isin(cfg.exclude_categories).to_numpy()[:, None]

    ii, tt = np.nonzero(valid)
    return pd.DataFrame({
        "customer_id": keys["customer_id"].to_numpy()[ii], "category": keys["category"].to_numpy()[ii],
        "month": months[tt], "spend": S[ii, tt], "expected": expected[ii, tt], "avg_ref": M[ii, tt],
        "excess": excess[ii, tt], "z": z[ii, tt], "adj_ratio": adj_ratio[ii, tt],
        "peer_ratio": peer[ii, tt], "elev_count": elev_count[ii, tt],
        "z_shift": z_shift[ii, tt], "ratio_shift": ratio_shift[ii, tt], "count_ratio": count_ratio[ii, tt],
        "flag": flag[ii, tt],
    })


def detect(grid: pd.DataFrame, cfg: DetectorConfig = None, cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flagged cells -> budgeted, ranked alerts with messages."""
    cfg = BALANCED if cfg is None else cfg
    cells = score_cells(grid, cfg) if cells is None else cells
    a = cells[cells["flag"]].copy()
    if a.empty:
        return a.assign(rank=[], severity=[], message=[])
    a["rank"] = a.groupby(["customer_id", "month"])["excess"].rank(method="first", ascending=False).astype(int)
    if cfg.max_alerts is not None:
        a = a[a["rank"] <= cfg.max_alerts]
    a["severity"] = np.where(a["z"] >= cfg.spike_z, "spike", "sustained")
    a["message"] = [_message(r, cfg) for r in a.itertuples()]
    return a.sort_values(["customer_id", "month", "rank"]).reset_index(drop=True)


def _message(r, cfg: DetectorConfig) -> str:
    pct = (r.spend / r.avg_ref - 1) * 100 if r.avg_ref > 0 else float("nan")
    msg = (f"{r.category} spending is {pct:.0f}% higher than your {cfg.window}-month average "
           f"(Rs {r.spend:,.0f} vs Rs {r.avg_ref:,.0f})")
    if cfg.shift_len > 1 and r.z < cfg.spike_z and r.z_shift >= cfg.shift_z:
        msg += f"; the last {cfg.shift_len} months average {(r.ratio_shift - 1) * 100:.0f}% above normal, so this looks like a sustained shift"
    elif cfg.use_persistence and r.elev_count >= cfg.persist_m and r.z < cfg.spike_z:
        msg += f"; it has been elevated in {int(r.elev_count)} of the last {cfg.persist_n} months"
    elif r.z >= cfg.spike_z:
        msg += "; this looks like a one-off spike"
    if cfg.peer_adjust and abs(r.peer_ratio - 1) >= 0.15:
        msg += (f". Similar customers' {r.category.lower()} spending moved {(r.peer_ratio - 1) * 100:+.0f}% "
                f"this month, so part of this is seasonal")
    return msg + "."


def with_(cfg: DetectorConfig, **kw) -> DetectorConfig:
    return replace(cfg, **kw)
