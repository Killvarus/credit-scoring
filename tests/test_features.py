import numpy as np
import pandas as pd
from sklearn.base import clone

from credit_scoring.data import TARGET, clean
from credit_scoring.features import WoEBinner, add_features
from credit_scoring.models import ENGINEERED, zoo


def test_engineered_features_are_finite(frame):
    out = add_features(frame)
    assert set(ENGINEERED) <= set(out.columns)
    assert np.isfinite(out[ENGINEERED].to_numpy(float)).all()


def test_add_features_is_rowwise(frame):
    # Признак строки не зависит от остальных строк — значит, утечки train → test нет
    full = add_features(frame)
    part = add_features(frame.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], part)


def test_months_since_delay_and_pay_ratio():
    row = add_features(clean(pd.DataFrame([{
        "LIMIT_BAL": 1000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_1": 0, "PAY_2": 0, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 500, "BILL_AMT2": 0, "BILL_AMT3": 400, "BILL_AMT4": 400, "BILL_AMT5": 400, "BILL_AMT6": 400,
        "PAY_AMT1": 0, "PAY_AMT2": 200, "PAY_AMT3": 100, "PAY_AMT4": 100, "PAY_AMT5": 100, "PAY_AMT6": 100,
    }]))).iloc[0]
    assert row["months_since_delay"] == 2  # PAY_3 — третий месяц с конца, индекс 2
    assert row["delay_max"] == 2 and row["delay_count"] == 1
    assert row["pay_ratio_1"] == 1.0       # счёт прошлого месяца нулевой — считаем долг погашенным
    assert np.isclose(row["pay_ratio_mean"], (1.0 + 0.5 + 0.25 + 0.25 + 0.25) / 5)


def test_clean_merges_undocumented_categories():
    frame = pd.DataFrame({"EDUCATION": [0, 1, 5, 6], "MARRIAGE": [0, 1, 2, 3]})
    out = clean(frame)
    assert out["EDUCATION"].tolist() == [4, 1, 4, 4]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3]


def test_woe_sign_and_iv():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-2 * x))).astype(int)  # больше x → чаще «плохой»
    binner = WoEBinner(n_bins=5).fit(pd.DataFrame({"x": x, "noise": rng.normal(size=5000)}), y)
    woe = binner.woe_["x"]
    assert np.all(np.diff(woe) < 0)  # WoE = ln(good/bad) убывает с риском
    assert binner.iv_["x"] > 0.5 > binner.iv_["noise"]
    out = binner.transform(pd.DataFrame({"x": [-100.0, 100.0], "noise": [0.0, 0.0]}))
    assert out["x"].iloc[0] == woe[0] and out["x"].iloc[1] == woe[-1]  # значения вне train → крайние бины


def test_all_models_fit_and_predict(frame):
    data = add_features(frame)
    for name, model in zoo().items():
        model = clone(model)
        if name == "KNN":
            model.set_params(clf__n_neighbors=10)
        p = model.fit(data, data[TARGET]).predict_proba(data)[:, 1]
        assert p.shape == (len(data),) and np.all((p >= 0) & (p <= 1)), name
