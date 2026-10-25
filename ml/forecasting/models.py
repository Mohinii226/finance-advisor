"""
Forecasters. Every model implements ONE method:

    forecast(Y, months, meta) -> np.ndarray [n_series]    # prediction for the month AFTER the last column of Y

`Y` only contains history up to the forecast origin, so a model cannot see the future (global models fit inside
`forecast` using that history alone). Models whose library is missing report `available() == False`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1.0


class Forecaster:
    name = "base"

    @classmethod
    def available(cls) -> bool:
        return True

    def forecast(self, Y: np.ndarray, months: pd.DatetimeIndex, meta: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------ simple baselines (tested)
class NaiveLast(Forecaster):
    name = "naive_last"

    def forecast(self, Y, months, meta):
        return Y[:, -1].copy()


class MovingAverage(Forecaster):
    def __init__(self, k: int):
        self.k, self.name = k, f"mean{k}"

    def forecast(self, Y, months, meta):
        return Y[:, -self.k:].mean(axis=1)


class MedianK(Forecaster):
    def __init__(self, k: int = 6):
        self.k, self.name = k, f"median{k}"

    def forecast(self, Y, months, meta):
        return np.median(Y[:, -self.k:], axis=1)


class SeasonalNaive(Forecaster):
    """Same month last year (needs 12 months of history; falls back to the 6-month mean before that)."""
    name = "seasonal_naive"

    def forecast(self, Y, months, meta):
        return Y[:, -12].copy() if Y.shape[1] >= 12 else Y[:, -6:].mean(axis=1)


class SimpleExpSmoothing(Forecaster):
    """SES with alpha chosen per series by minimising in-sample one-step squared error."""
    name = "ses"
    ALPHAS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

    def forecast(self, Y, months, meta):
        n, T = Y.shape
        level = np.repeat(Y[:, [0]], len(self.ALPHAS), axis=1)           # [n, n_alpha]
        sse = np.zeros_like(level)
        for t in range(1, T):
            sse += (Y[:, [t]] - level) ** 2
            level = self.ALPHAS * Y[:, [t]] + (1 - self.ALPHAS) * level
        best = sse.argmin(axis=1)
        return level[np.arange(n), best]


# ------------------------------------------------------------------ global gradient boosting (tested)
def _features(Y: np.ndarray, j: int, meta: pd.DataFrame, months: pd.DatetimeIndex | None, month_of_target: int) -> tuple[np.ndarray, np.ndarray]:
    """Features for predicting column j from columns < j. Returns (X, scale)."""
    hist = Y[:, :j]
    scale = hist[:, -6:].mean(axis=1) + EPS
    cols = [hist[:, -l] / scale for l in range(1, 13)]
    r = lambda k: hist[:, -k:]
    cols += [r(3).mean(1) / scale, r(6).mean(1) / scale, r(12).mean(1) / scale, r(6).std(1) / scale,
             r(6).min(1) / scale, r(6).max(1) / scale, np.log(scale),
             meta["monthly_income"].to_numpy() / scale, meta["income_type_code"].to_numpy(),
             np.full(len(scale), month_of_target), np.sin(2 * np.pi * month_of_target / 12) * np.ones(len(scale)),
             np.cos(2 * np.pi * month_of_target / 12) * np.ones(len(scale))]
    return np.column_stack(cols), scale


class GlobalGBM(Forecaster):
    """One gradient-boosting model for ALL series, predicting next month's spend as a ratio of the recent 6-month level.
    Pooling customers is what lets a 36-point series benefit from seasonality seen across customers."""

    def __init__(self, loss: str = "absolute_error", max_iter: int = 150, learning_rate: float = 0.05, seed: int = 0):
        self.loss, self.max_iter, self.lr, self.seed = loss, max_iter, learning_rate, seed
        self.name = "gbm_l1" if loss == "absolute_error" else "gbm_l2"

    def forecast(self, Y, months, meta):
        from sklearn.ensemble import HistGradientBoostingRegressor
        n, T = Y.shape
        moy = months.month.to_numpy()
        Xs, ys = [], []
        for j in range(12, T):                                   # targets whose features AND target lie inside the history
            X, s = _features(Y, j, meta, months, int(moy[j]))
            Xs.append(X); ys.append(Y[:, j] / s)
        if not Xs:
            return Y[:, -6:].mean(axis=1)
        X, y = np.vstack(Xs), np.concatenate(ys)
        y = np.clip(y, 0, 6)                                     # winsorise ratio targets so one disaster does not dominate
        model = HistGradientBoostingRegressor(loss=self.loss, max_iter=self.max_iter, learning_rate=self.lr, max_leaf_nodes=15,
                                              min_samples_leaf=30, l2_regularization=1.0, random_state=self.seed)
        model.fit(X, y)
        next_month = int((months[-1] + pd.DateOffset(months=1)).month)
        Xn, sn = _features(np.hstack([Y, np.zeros((n, 1))]), T, meta, months, next_month)
        return np.clip(model.predict(Xn), 0.2, 5.0) * sn


# ------------------------------------------------------------------ optional models (written for local runs; NOT run in my sandbox)
class ARIMAModel(Forecaster):
    """Per-series ARIMA; picks the order with the lowest AICc from a small grid. Requires statsmodels."""
    name = "arima"
    ORDERS = [(0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), (2, 0, 0)]

    @classmethod
    def available(cls) -> bool:
        try:
            import statsmodels  # noqa: F401
            return True
        except Exception:
            return False

    def forecast(self, Y, months, meta):
        from statsmodels.tsa.arima.model import ARIMA
        out = np.empty(len(Y))
        for i, y in enumerate(Y):
            best, best_aicc = None, np.inf
            for order in self.ORDERS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = ARIMA(y, order=order, trend="c" if order[1] == 0 else "n").fit()
                    k, n = len(res.params), len(y)
                    aicc = res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
                    if aicc < best_aicc:
                        best, best_aicc = res, aicc
                except Exception:
                    continue
            out[i] = max(float(best.forecast(1)[0]), 0.0) if best is not None else float(y[-6:].mean())
        return out


class ProphetModel(Forecaster):
    """Per-series Prophet (yearly seasonality only when >= 24 months). Slow: use --max-series. Requires prophet."""
    name = "prophet"

    @classmethod
    def available(cls) -> bool:
        try:
            import prophet  # noqa: F401
            return True
        except Exception:
            return False

    def forecast(self, Y, months, meta):
        import logging
        from prophet import Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        out = np.empty(len(Y))
        for i, y in enumerate(Y):
            df = pd.DataFrame({"ds": months, "y": y})
            try:
                m = Prophet(yearly_seasonality=len(y) >= 24, weekly_seasonality=False, daily_seasonality=False)
                m.fit(df)
                nxt = pd.DataFrame({"ds": [months[-1] + pd.DateOffset(months=1)]})
                out[i] = max(float(m.predict(nxt)["yhat"].iloc[0]), 0.0)
            except Exception:
                out[i] = float(y[-6:].mean())
        return out


class LSTMGlobal(Forecaster):
    """One small LSTM for all series on 12-month windows of level-normalised spend (+ month-of-year). Requires torch."""
    name = "lstm"

    def __init__(self, hidden: int = 32, epochs: int = 60, lr: float = 0.01, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    @classmethod
    def available(cls) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _windows(Y, moy, j):
        """Input sequence for predicting column j: the 12 columns before it, scaled by their own 6-month level."""
        hist = Y[:, j - 12:j]
        scale = Y[:, j - 6:j].mean(axis=1, keepdims=True) + EPS
        m = moy[j - 12:j]
        feats = np.stack([hist / scale, np.broadcast_to(np.sin(2 * np.pi * m / 12), hist.shape),
                          np.broadcast_to(np.cos(2 * np.pi * m / 12), hist.shape)], axis=-1)       # [n, 12, 3]
        return feats.astype("float32"), scale[:, 0]

    def forecast(self, Y, months, meta):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        n, T = Y.shape
        if T < 13:
            return Y[:, -6:].mean(axis=1)
        moy = np.append(months.month.to_numpy(), (months[-1] + pd.DateOffset(months=1)).month)
        Yx = np.hstack([Y, np.zeros((n, 1))])
        Xs, ys = [], []
        for j in range(12, T):
            f, s = self._windows(Yx, moy, j)
            Xs.append(f); ys.append(np.clip(Y[:, j] / s, 0, 6).astype("float32"))
        X, y = torch.tensor(np.concatenate(Xs)), torch.tensor(np.concatenate(ys))

        class Net(nn.Module):
            def __init__(s, h):
                super().__init__()
                s.lstm, s.head = nn.LSTM(3, h, batch_first=True), nn.Linear(h, 1)

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.head(o[:, -1]).squeeze(-1)

        net = Net(self.hidden)
        opt, loss_fn = torch.optim.Adam(net.parameters(), lr=self.lr), nn.L1Loss()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for b in range(0, len(X), 256):
                idx = perm[b:b + 256]
                opt.zero_grad()
                loss_fn(net(X[idx]), y[idx]).backward()
                opt.step()
        net.eval()
        f, s = self._windows(Yx, moy, T)
        with torch.no_grad():
            pred = net(torch.tensor(f)).numpy()
        return np.clip(pred, 0.2, 5.0) * s


def default_models(include_optional: bool = True) -> list[Forecaster]:
    models: list[Forecaster] = [NaiveLast(), MovingAverage(3), MovingAverage(6), MedianK(6), SeasonalNaive(), SimpleExpSmoothing(),
                                GlobalGBM("squared_error"), GlobalGBM("absolute_error")]
    if include_optional:
        models += [m for m in (ARIMAModel(), ProphetModel(), LSTMGlobal()) if m.available()]
    return models
