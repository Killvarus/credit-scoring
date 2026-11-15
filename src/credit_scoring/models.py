"""
Набор моделей для сравнения. Каждая модель — sklearn-совместимый пайплайн,
который принимает DataFrame после add_features и сам выбирает нужные колонки.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import BILL_COLS, CATEGORICAL, PAY_COLS, PAYAMT_COLS
from .features import WoEBinner

RAW_NUM = ["LIMIT_BAL", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]
ENGINEERED = [
    *[f"util_{i}" for i in range(1, 7)], "util_mean", "util_max", "util_trend",
    "pay_ratio_1", "pay_ratio_mean", "pay_ratio_min",
    "delay_max", "delay_count", "delay_recent2", "months_since_delay",
    "n_revolving", "n_paid_full", "n_no_usage",
    "spend_mean_rel", "pay_mean_rel", "log_limit",
]
AMOUNTS = ["LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS]
ALL_FEATURES = [*RAW_NUM, *CATEGORICAL, *ENGINEERED]
SEED = 42


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _linear_preprocessor(engineered: bool) -> ColumnTransformer:
    num = [c for c in RAW_NUM if c not in AMOUNTS] + (ENGINEERED if engineered else [])
    parts = [
        ("amounts", make_pipeline(FunctionTransformer(signed_log1p), StandardScaler()), AMOUNTS),
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
    if engineered:
        # Статус платежа нелинеен: -2/-1/0 почти одинаково безопасны, 1 и 2 резко опаснее
        parts.append(("status", OneHotEncoder(handle_unknown="ignore", min_frequency=50), ["PAY_1", "PAY_2"]))
    return ColumnTransformer(parts)


def logreg(engineered: bool = False, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", _linear_preprocessor(engineered)),
        ("clf", LogisticRegression(C=C, max_iter=5000)),
    ])


SCORECARD_FEATURES = [
    "PAY_1", "PAY_2", "PAY_3", "delay_max", "delay_count", "months_since_delay", "LIMIT_BAL",
    "util_1", "util_mean", "pay_ratio_1", "pay_ratio_mean", "pay_mean_rel", "n_paid_full",
    "n_no_usage", "AGE", "EDUCATION",
]


class Scorecard(ClassifierMixin, BaseEstimator):
    """WoE + логистическая регрессия с контролем знаков — интерпретируемая модель для балльной карты.

    WoE растёт с надёжностью клиента, поэтому все коэффициенты должны быть отрицательными (модель
    предсказывает дефолт). Положительный коэффициент — признак мультиколлинеарности: признак дублирует
    другой и «исправляет» его. Такие признаки удаляются по одному (самый положительный первым),
    пока все знаки не станут правильными. Признаки с IV < min_iv отбрасываются сразу.
    """

    def __init__(self, features=tuple(SCORECARD_FEATURES), n_bins: int = 10, min_iv: float = 0.02, C: float = 1.0):
        self.features = features
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y)
        self.woe_ = WoEBinner(n_bins=self.n_bins).fit(X[list(self.features)], y)
        W = self.woe_.transform(X[list(self.features)])
        kept = [f for f in self.features if self.woe_.iv_[f] >= self.min_iv]
        self.dropped_ = [(f, "IV") for f in self.features if f not in kept]
        while True:
            self.clf_ = LogisticRegression(C=self.C, max_iter=5000).fit(W[kept], y)
            coef = self.clf_.coef_[0]
            if coef.max() <= 0 or len(kept) == 1:
                break
            worst = kept[int(coef.argmax())]
            self.dropped_.append((worst, f"знак {coef.max():+.2f}"))
            kept.remove(worst)
        self.features_ = kept
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.woe_.transform(X[list(self.features)])[self.features_])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def scorecard() -> Scorecard:
    return Scorecard()


def knn(k: int = 100) -> Pipeline:
    return Pipeline([
        ("prep", _linear_preprocessor(engineered=True)),
        ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)),
    ])


def _select(columns):
    return ColumnTransformer([("keep", "passthrough", columns)], verbose_feature_names_out=False).set_output(transform="pandas")


def random_forest() -> Pipeline:
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=SEED)
    return Pipeline([("select", _select(ALL_FEATURES)), ("clf", clf)])


LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=80,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
)


def lightgbm(**params) -> Pipeline:
    clf = LGBMClassifier(**{**LGBM_PARAMS, **params}, random_state=SEED, verbose=-1, n_jobs=4)
    return Pipeline([("select", _select(ALL_FEATURES)), ("clf", clf)])


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """CatBoost не проходит sklearn.clone (меняет cat_features в конструкторе) — создаём модель в fit."""

    def __init__(self, iterations: int = 800, learning_rate: float = 0.03, depth: int = 5, l2_leaf_reg: float = 5):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            cat_features=[c for c in CATEGORICAL if c in X.columns], random_seed=SEED, verbose=0, thread_count=4,
            allow_writing_files=False,
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def catboost() -> Pipeline:
    return Pipeline([("select", _select(ALL_FEATURES)), ("clf", CatBoostWrapper())])


def xgboost() -> Pipeline:
    clf = XGBClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, n_jobs=4,
    )
    return Pipeline([("select", _select(ALL_FEATURES)), ("clf", clf)])


def zoo() -> dict[str, Pipeline]:
    return {
        "dummy (доля дефолтов)": DummyClassifier(strategy="prior"),
        "logreg (сырые признаки) — бейзлайн": logreg(engineered=False),
        "logreg + новые признаки": logreg(engineered=True),
        "WoE-скоркарта": scorecard(),
        "KNN": knn(),
        "random forest": random_forest(),
        "LightGBM": lightgbm(),
        "CatBoost": catboost(),
        "XGBoost": xgboost(),
    }
