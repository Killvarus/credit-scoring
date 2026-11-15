import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import BILL_COLS, PAY_COLS, PAYAMT_COLS, TARGET


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Синтетические клиенты с той же схемой, что у UCI: риск растёт с просрочкой и утилизацией."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([10_000, 50_000, 100_000, 200_000, 500_000], n)
    frame = pd.DataFrame({
        "LIMIT_BAL": limit,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    })
    for c in PAY_COLS:
        frame[c] = rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], n)
    for c in BILL_COLS:
        frame[c] = (limit * rng.uniform(-0.05, 1.0, n)).round()
    for c in PAYAMT_COLS:
        frame[c] = (limit * rng.uniform(0, 0.2, n)).round()
    logit = -2.0 + 0.9 * frame["PAY_1"].clip(0, None) + 1.0 * frame["BILL_AMT1"] / limit
    frame[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
