"""
Метрики скоринга.

ROC-AUC  — вероятность, что случайный дефолтник получит PD выше случайного «хорошего».
Gini     — 2·AUC − 1, так метрику обычно называют в банках.
KS       — максимум разницы накопленных распределений PD у «плохих» и «хороших».
PR-AUC   — средняя точность; чувствительна к доле положительных (здесь 22%).
Brier, LogLoss — качество самих вероятностей (калибровка + разделение).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def score(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
    }


def bootstrap_auc(y, p, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% доверительный интервал ROC-AUC перевыборкой тестовых клиентов."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(y, p_a, p_b, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """Разница AUC(b) − AUC(a) на одних и тех же перевыборках: ДИ и доля случаев, где b не лучше."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    deltas = np.asarray(deltas)
    return {
        "delta": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float((deltas <= 0).mean()),
    }


def profit_curve(y, p, gain_good: float = 1.0, loss_bad: float = 5.0, grid: int = 101):
    """Прибыль при одобрении доли клиентов с наименьшим PD.

    Модель экономики условная: одобренный «хороший» клиент приносит gain_good,
    одобренный дефолтник уносит loss_bad (в тех же единицах). Возвращает доли одобрения и прибыль
    на одного заявителя.
    """
    y = np.asarray(y)
    order = np.argsort(np.asarray(p), kind="stable")
    bad_sorted = y[order]
    cum_bad = np.concatenate([[0], np.cumsum(bad_sorted)])
    cum_good = np.arange(len(y) + 1) - cum_bad
    rates = np.linspace(0, 1, grid)
    k = np.round(rates * len(y)).astype(int)
    profit = (gain_good * cum_good[k] - loss_bad * cum_bad[k]) / len(y)
    return rates, profit
