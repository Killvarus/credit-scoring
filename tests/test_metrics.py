import numpy as np

from credit_scoring.ab import run_policy_test, sample_size_per_arm, two_proportion_ztest
from credit_scoring.metrics import ks_statistic, paired_bootstrap_delta, profit_curve, score


def test_score_relations():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 1000), 0, 1)
    m = score(y, p)
    assert np.isclose(m["gini"], 2 * m["roc_auc"] - 1)
    assert 0 < m["ks"] <= 1


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, np.arange(100)) == 1.0
    assert ks_statistic(y, np.ones(100)) == 0.0


def test_profit_curve_bounds():
    y = np.array([0, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
    rates, profit = profit_curve(y, p, gain_good=1, loss_bad=5, grid=6)
    assert profit[0] == 0
    assert np.isclose(profit[-1], (3 - 10) / 5)          # одобрили всех
    assert np.isclose(profit.max(), 3 / 5)               # одобрили ровно трёх хороших


def test_paired_bootstrap_identical_models():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 300), rng.random(300)
    d = paired_bootstrap_delta(y, p, p, n=50)
    assert d["delta"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_ztest_and_sample_size():
    z, p = two_proportion_ztest(100, 1000, 100, 1000)
    assert z == 0 and np.isclose(p, 1)
    assert two_proportion_ztest(150, 1000, 100, 1000)[1] < 0.01
    assert sample_size_per_arm(0.12, 0.10) > sample_size_per_arm(0.12, 0.08)


def test_aa_test_false_positive_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    score_ = rng.random(4000)
    pvals = np.array([run_policy_test(y, score_, score_, 0.7, rng)["p_value"] for _ in range(400)])
    assert 0.02 < (pvals < 0.05).mean() < 0.09
