"""
Статистика для A/B-теста кредитной политики.

Сравниваем долю дефолтов среди одобренных (bad rate) в двух группах при одинаковой доле одобрений.
Фиксировать долю одобрений обязательно: иначе bad rate «улучшается» простым ужесточением политики.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def two_proportion_ztest(bad_a: int, n_a: int, bad_b: int, n_b: int) -> tuple[float, float]:
    """Двусторонний z-тест разности долей. Возвращает (z, p-value)."""
    p_a, p_b = bad_a / n_a, bad_b / n_b
    pooled = (bad_a + bad_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    return float(z), float(2 * norm.sf(abs(z)))


def sample_size_per_arm(p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Сколько одобренных клиентов нужно в каждой группе, чтобы заметить разницу p_a → p_b."""
    z_a, z_b = norm.isf(alpha / 2), norm.isf(1 - power)
    p_bar = (p_a + p_b) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) ** 2
    return int(np.ceil(num / (p_a - p_b) ** 2))


def run_policy_test(y, score_a, score_b, approval_rate: float, rng: np.random.Generator) -> dict[str, float]:
    """Одна рандомизация: клиенты делятся пополам, в каждой группе своя модель одобряет approval_rate лучших."""
    y = np.asarray(y)
    arm = rng.random(len(y)) < 0.5
    result = {}
    for name, mask, score in (("a", ~arm, np.asarray(score_a)), ("b", arm, np.asarray(score_b))):
        idx = np.flatnonzero(mask)
        k = int(round(approval_rate * len(idx)))
        approved = idx[np.argsort(score[idx], kind="stable")[:k]]
        result[f"n_{name}"] = k
        result[f"bad_{name}"] = int(y[approved].sum())
    result["rate_a"] = result["bad_a"] / result["n_a"]
    result["rate_b"] = result["bad_b"] / result["n_b"]
    result["z"], result["p_value"] = two_proportion_ztest(result["bad_a"], result["n_a"], result["bad_b"], result["n_b"])
    return result
