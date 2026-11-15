"""
Признаки и WoE-биннинг.

add_features — построчные признаки (не обучаются на данных, утечки между train и test нет):
  утилизация лимита, доля погашения счёта, статистики просрочек, тренды счёта и трат.
WoEBinner — классический для скоркарт перевод признака в Weight of Evidence по квантильным
  бинам; обучается только на train-фолде.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import BILL_COLS, PAY_COLS, PAYAMT_COLS

EPS = 1.0


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    limit = out["LIMIT_BAL"].astype(float)
    bills = out[BILL_COLS].to_numpy(float)
    pays = out[PAYAMT_COLS].to_numpy(float)
    status = out[PAY_COLS].to_numpy(float)

    # Утилизация: какая доля лимита занята счётом (1 — сентябрь, 6 — апрель)
    util = bills / limit.to_numpy()[:, None]
    for i in range(6):
        out[f"util_{i + 1}"] = util[:, i]
    out["util_mean"] = util.mean(1)
    out["util_max"] = util.max(1)
    out["util_trend"] = util[:, 0] - util[:, 5]

    # Доля погашения: платёж месяца i закрывает счёт предыдущего месяца i+1
    prev_bill = np.clip(bills[:, 1:], EPS, None)
    pay_ratio = np.clip(pays[:, :5] / prev_bill, 0, 2)
    pay_ratio[bills[:, 1:] <= 0] = 1.0  # нечего платить — считаем, что долг погашен
    out["pay_ratio_1"] = pay_ratio[:, 0]
    out["pay_ratio_mean"] = pay_ratio.mean(1)
    out["pay_ratio_min"] = pay_ratio.min(1)

    # Просрочки: статус ≥ 1 — просрочка на столько месяцев
    delay = np.clip(status, 0, None)
    out["delay_max"] = delay.max(1)
    out["delay_count"] = (status >= 1).sum(1)
    out["delay_recent2"] = delay[:, :2].sum(1)
    # Сколько месяцев назад была последняя просрочка (7 — не было)
    has_delay = status >= 1
    out["months_since_delay"] = np.where(has_delay.any(1), has_delay.argmax(1), 7)
    out["n_revolving"] = (status == 0).sum(1)
    out["n_paid_full"] = (status == -1).sum(1)
    out["n_no_usage"] = (status == -2).sum(1)

    # Новые траты за месяц: изменение счёта плюс внесённый платёж
    spend = bills[:, :5] - bills[:, 1:] + pays[:, :5]
    out["spend_mean_rel"] = spend.mean(1) / limit
    out["pay_mean_rel"] = pays.mean(1) / limit
    out["log_limit"] = np.log(limit)
    return out


class WoEBinner(BaseEstimator, TransformerMixin):
    """Квантильные бины → WoE = ln(доля «хороших» в бине / доля «плохих» в бине).

    Положительный WoE — бин надёжнее среднего. Сглаживание +0.5 защищает от пустых бинов.
    Information Value (IV) каждого признака сохраняется в iv_.
    """

    def __init__(self, n_bins: int = 10, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.edges_, self.woe_, self.iv_ = {}, {}, {}
        n_bad, n_good = y.sum(), (1 - y).sum()
        for col in X.columns:
            values = X[col].to_numpy(float)
            edges = np.unique(np.quantile(values, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            bins = np.searchsorted(edges, values, side="right")
            k = len(edges) + 1
            bad = np.bincount(bins, weights=y, minlength=k) + self.smoothing
            good = np.bincount(bins, weights=1 - y, minlength=k) + self.smoothing
            dist_bad, dist_good = bad / (n_bad + k * self.smoothing), good / (n_good + k * self.smoothing)
            woe = np.log(dist_good / dist_bad)
            self.edges_[col], self.woe_[col] = edges, woe
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = {}
        for col in self.columns_:
            bins = np.searchsorted(self.edges_[col], X[col].to_numpy(float), side="right")
            out[col] = self.woe_[col][bins]
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)
