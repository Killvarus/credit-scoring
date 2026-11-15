"""
Подбор, сравнение и итоговая оценка моделей.

1. Подбор k для KNN и случайный поиск гиперпараметров LightGBM — 5-fold CV на train.
2. Сравнение всех моделей на той же 5-fold CV (ROC-AUC, среднее ± std).
3. Обучение на всём train, метрики на отложенном test, бутстрэп-интервалы.
4. Графики: ROC, калибровка, кривая прибыли, вклад признаков (TreeSHAP), балльная карта.

    python scripts/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring import models as M  # noqa: E402
from credit_scoring.data import TARGET, load, split  # noqa: E402
from credit_scoring.features import add_features  # noqa: E402
from credit_scoring.metrics import bootstrap_auc, paired_bootstrap_delta, profit_curve, score  # noqa: E402

REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
for d in (FIG, MODELS, ROOT / "data"):
    d.mkdir(parents=True, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=M.SEED)
BASELINE = "logreg (сырые признаки) — бейзлайн"


def cv_auc(model, X, y) -> tuple[float, float, np.ndarray]:
    from sklearn.metrics import roc_auc_score

    oof = cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]
    folds = [roc_auc_score(y.iloc[te], oof[te]) for _, te in CV.split(X, y)]
    return float(np.mean(folds)), float(np.std(folds)), oof


def tune(X, y) -> tuple[int, dict, list[str]]:
    log = ["## Подбор гиперпараметров (5-fold CV на train)", ""]
    knn_scores = {k: cv_auc(M.knn(k), X, y)[0] for k in (25, 50, 100, 200, 400)}
    best_k = max(knn_scores, key=knn_scores.get)
    log += ["KNN, ROC-AUC по числу соседей: " + ", ".join(f"k={k}: {v:.4f}" for k, v in knn_scores.items()), ""]

    rng = np.random.default_rng(M.SEED)
    grid = []
    for _ in range(16):
        params = dict(
            learning_rate=float(rng.choice([0.01, 0.02, 0.05])),
            num_leaves=int(rng.choice([7, 15, 31, 63])),
            min_child_samples=int(rng.choice([20, 50, 100, 200])),
            colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])),
            reg_lambda=float(rng.choice([0.0, 1.0, 5.0, 20.0])),
        )
        params["n_estimators"] = int({0.01: 1200, 0.02: 600, 0.05: 250}[params["learning_rate"]])
        grid.append((cv_auc(M.lightgbm(**params), X, y)[0], params))
    grid.sort(key=lambda t: -t[0])
    log += ["LightGBM, случайный поиск (16 конфигураций), лучшие три:", ""]
    log += [f"- AUC {a:.4f}: {p}" for a, p in grid[:3]]
    log += [f"- худшая: AUC {grid[-1][0]:.4f} — разброс между конфигурациями мал, данные важнее параметров", ""]
    return best_k, grid[0][1], log


def scorecard_table(pipe, pdo: float = 20, base_score: float = 600, base_odds: float = 50) -> pd.DataFrame:
    """Балльная карта: баллы за бин каждого признака; сумма баллов растёт с надёжностью клиента."""
    woe, clf = pipe.woe_, pipe.clf_
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    n = len(pipe.features_)
    rows = []
    for coef, col in zip(clf.coef_[0], pipe.features_):
        edges = np.concatenate([[-np.inf], woe.edges_[col], [np.inf]])
        for i, w in enumerate(woe.woe_[col]):
            points = (offset - factor * clf.intercept_[0]) / n - factor * coef * w
            rows.append({"признак": col, "от": edges[i], "до": edges[i + 1], "WoE": w, "баллы": points})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    train, test = split(load())
    train, test = add_features(train), add_features(test)
    X, y, Xt, yt = train, train[TARGET], test, test[TARGET]

    best_k, lgbm_params, log = tune(X, y)
    zoo = M.zoo()
    zoo["KNN"] = M.knn(best_k)
    zoo["LightGBM"] = M.lightgbm(**lgbm_params)

    cv_rows, test_rows, preds = [], [], {}
    for name, model in zoo.items():
        start = time.time()
        mean, std, _ = cv_auc(model, X, y)
        model.fit(X, y)
        p = model.predict_proba(Xt)[:, 1]
        preds[name] = p
        lo, hi = bootstrap_auc(yt, p, n=500)
        cv_rows.append({"модель": name, "cv_auc": mean, "cv_std": std})
        test_rows.append({"модель": name, **score(yt, p), "auc_ci": f"{lo:.3f}–{hi:.3f}", "сек": time.time() - start})
        print(f"{name:40s} CV {mean:.4f}±{std:.4f}  test {test_rows[-1]['roc_auc']:.4f}", flush=True)

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    # Лучшую модель выбираем по CV, а не по test, чтобы не подгонять выбор под test
    best = cv_df.sort_values("cv_auc", ascending=False).iloc[0]["модель"]
    delta = paired_bootstrap_delta(yt, preds[BASELINE], preds[best], n=1000)
    delta_fe = paired_bootstrap_delta(yt, preds[BASELINE], preds["logreg + новые признаки"], n=1000)

    # Прибыль: одобренный «хороший» = +1, одобренный дефолтник = −5 (условные единицы)
    profit = {}
    for name in (BASELINE, "WoE-скоркарта", best):
        rates, prof = profit_curve(yt, preds[name])
        i = int(np.argmax(prof))
        profit[name] = {"approval_rate": float(rates[i]), "profit_per_applicant": float(prof[i]),
                        "bad_rate_approved": float(yt.to_numpy()[np.argsort(preds[name])][: int(rates[i] * len(yt))].mean())}

    # ---------- графики
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    show = [BASELINE, "logreg + новые признаки", "WoE-скоркарта", "KNN", "random forest", best]
    for name in dict.fromkeys(show):
        fpr, tpr, _ = roc_curve(yt, preds[name])
        axes[0].plot(fpr, tpr, label=f"{name.split(' —')[0]} ({score(yt, preds[name])['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(title="ROC на test", xlabel="FPR", ylabel="TPR"); axes[0].legend(fontsize=8)
    for name in (BASELINE, "WoE-скоркарта", "KNN", best):
        frac, mean_p = calibration_curve(yt, preds[name], n_bins=10, strategy="quantile")
        axes[1].plot(mean_p, frac, "o-", ms=3, label=name.split(" —")[0])
    axes[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[1].set(title="Калибровка (10 квантильных бинов)", xlabel="предсказанный PD", ylabel="доля дефолтов"); axes[1].legend(fontsize=8)
    for name in profit:
        rates, prof = profit_curve(yt, preds[name])
        axes[2].plot(rates, prof, label=name.split(" —")[0])
    axes[2].axhline(0, color="grey", lw=1)
    axes[2].set(title="Прибыль: +1 за хорошего, −5 за дефолт", xlabel="доля одобренных"); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=110)

    # TreeSHAP через pred_contrib у LightGBM — без внешних зависимостей
    lgbm = zoo["LightGBM"]
    Xs = lgbm.named_steps["select"].transform(Xt)
    contrib = lgbm.named_steps["clf"].booster_.predict(Xs, pred_contrib=True)[:, :-1]
    shap_imp = pd.Series(np.abs(contrib).mean(0), index=Xs.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    shap_imp.head(15)[::-1].plot.barh(ax=ax, color="#4a78b5")
    ax.set_title("LightGBM: средний |SHAP| на test (логит)"); fig.tight_layout(); fig.savefig(FIG / "shap_importance.png", dpi=110)

    card = scorecard_table(zoo["WoE-скоркарта"])
    sc = zoo["WoE-скоркарта"]
    coefs = pd.Series(sc.clf_.coef_[0], index=sc.features_)

    # ---------- артефакты
    joblib.dump(zoo[best], MODELS / "best_model.joblib")
    pd.DataFrame({"y": yt.to_numpy(), **{k: v for k, v in preds.items()}}).to_csv(ROOT / "data" / "test_predictions.csv", index=False)
    metrics = {
        "best_model": best, "lightgbm_params": lgbm_params, "knn_k": best_k,
        "cv": cv_rows, "test": test_df.drop(columns="сек").to_dict("records"),
        "delta_auc_best_vs_baseline": delta, "delta_auc_fe_vs_baseline": delta_fe, "profit": profit,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Результаты", "", *log, "## Сравнение на 5-fold CV и на отложенном test (6 000 клиентов)", "",
          "| модель | CV ROC-AUC | test ROC-AUC [95% ДИ] | Gini | KS | PR-AUC | Brier | LogLoss |", "|---|---|---|---|---|---|---|---|"]
    merged = test_df.merge(cv_df, on="модель")
    for _, r in merged.iterrows():
        md.append(f"| {r['модель']} | {r.cv_auc:.4f} ± {r.cv_std:.4f} | {r.roc_auc:.4f} [{r.auc_ci}] | {r.gini:.3f} | {r.ks:.3f} | "
                  f"{r.pr_auc:.3f} | {r.brier:.4f} | {r.logloss:.4f} |")
    md += ["", f"Лучшая по CV: **{best}**. Разница с бейзлайном на test (парный бутстрэп, 1000 повторов): "
           f"ΔAUC = {delta['delta']:+.4f} [{delta['ci_low']:+.4f}; {delta['ci_high']:+.4f}], p = {delta['p_value']:.3f}.",
           f"Новые признаки в логистической регрессии: ΔAUC = {delta_fe['delta']:+.4f} "
           f"[{delta_fe['ci_low']:+.4f}; {delta_fe['ci_high']:+.4f}].", "",
           "## Экономика (условная: +1 за одобренного хорошего, −5 за одобренного дефолтника)", "",
           "| модель | оптимальная доля одобрений | прибыль на заявителя | доля дефолтов среди одобренных |", "|---|---|---|---|"]
    md += [f"| {k} | {v['approval_rate']:.0%} | {v['profit_per_applicant']:.4f} | {v['bad_rate_approved']:.3f} |" for k, v in profit.items()]
    md += ["", "## Вклад признаков в LightGBM (средний |SHAP|, топ-10)", "", "| признак | средний модуль SHAP |", "|---|---|"]
    md += [f"| {k} | {v:.3f} |" for k, v in shap_imp.head(10).items()]
    md += ["", "## Балльная карта (WoE-скоркарта, PDO = 20, 600 баллов при шансах 50:1)", "",
           "Коэффициенты логистической регрессии при WoE (все отрицательные — чем выше WoE, тем ниже риск):", "",
           ", ".join(f"{k}: {v:.2f}" for k, v in coefs.items()), "",
           "Удалены (IV < 0.02 или неправильный знак из-за мультиколлинеарности, по порядку): "
           + ", ".join(f"{f} ({why})" for f, why in sc.dropped_), "",
           "Пример: баллы за PAY_1 и LIMIT_BAL (интервал [от; до), полная карта — reports/scorecard.csv)", "",
           "| признак | интервал | WoE | баллы |", "|---|---|---|---|"]
    for _, r in card[card["признак"].isin(["PAY_1", "LIMIT_BAL"])].iterrows():
        md.append(f"| {r['признак']} | [{r['от']:g}; {r['до']:g}) | {r['WoE']:+.2f} | {r['баллы']:.1f} |")
    md += ["", f"Время полного прогона: {time.time() - t0:.0f} с."]
    card.to_csv(REPORTS / "scorecard.csv", index=False)
    (REPORTS / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
