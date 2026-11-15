"""
Загрузка и очистка датасета Default of Credit Card Clients (UCI, Тайвань, 2005).

30 000 клиентов банка, 23 признака: лимит, пол, образование, семейное положение, возраст,
статусы платежей за 6 месяцев, суммы счетов и суммы платежей. Цель — дефолт
в следующем месяце (~22% положительных). Источник — зеркало UCI на OpenML (data_id=42477),
колонки там названы x1..x23 в порядке документации UCI.

Индекс 1 в PAY_/BILL_AMT/PAY_AMT — сентябрь 2005 (последний месяц), 6 — апрель.
Статус платежа PAY_i: -2 — не было потребления, -1 — оплачено полностью,
0 — внесён минимальный платёж (револьверный кредит), k ≥ 1 — просрочка на k месяцев.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OPENML_ID = 42477
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "credit_default.csv"
TARGET = "default"

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]


def download(path: Path = RAW_PATH) -> Path:
    """Скачивает датасет с OpenML и сохраняет CSV с понятными именами колонок."""
    if path.exists():
        return path
    from sklearn.datasets import fetch_openml

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(path.parent / "openml")).frame
    frame.columns = [*RAW_COLUMNS, TARGET]
    frame[TARGET] = frame[TARGET].astype(int)
    frame.to_csv(path, index=False)
    return path


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Сводит недокументированные категории к «другое»."""
    frame = frame.copy()
    # EDUCATION: в документации 1..4, в данных встречаются 0, 5, 6 — объединяем с «другое» (4)
    frame["EDUCATION"] = frame["EDUCATION"].where(frame["EDUCATION"].between(1, 4), 4)
    # MARRIAGE: документированы 1..3, значение 0 — тоже «другое» (3)
    frame["MARRIAGE"] = frame["MARRIAGE"].where(frame["MARRIAGE"].between(1, 3), 3)
    return frame


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    return clean(pd.read_csv(download(path)))


def split(frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированный отложенный test. EDA и подбор моделей делаются только на train."""
    from sklearn.model_selection import train_test_split

    return train_test_split(frame, test_size=test_size, stratify=frame[TARGET], random_state=seed)
