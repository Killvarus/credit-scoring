"""
EDA и проверка гипотез на train-части. Пишет reports/eda.md и графики в reports/figures/.

    python scripts/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.data import BILL_COLS, TARGET, load, split  # noqa: E402
from credit_scoring.features import WoEBinner, add_features  # noqa: E402
from credit_scoring.models import ALL_FEATURES  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def rate_by_quantile(frame: pd.DataFrame, col: str, q: int = 5) -> pd.Series:
    bins = pd.qcut(frame[col].rank(method="first"), q, labels=[f"Q{i}" for i in range(1, q + 1)])
    return frame.groupby(bins, observed=True)[TARGET].mean()


def main():
    train, _ = split(load())
    train = add_features(train)
    base = train[TARGET].mean()
    lines = ["# EDA (train, 24 000 клиентов)", "", f"Доля дефолтов: **{base:.3f}**.", ""]

    # Information Value всех признаков
    iv = pd.Series(WoEBinner(n_bins=10).fit(train[ALL_FEATURES], train[TARGET]).iv_).sort_values(ascending=False)
    lines += ["## Information Value (топ-15)", "",
              "IV < 0.02 — бесполезен, 0.02–0.1 — слабый, 0.1–0.3 — средний, > 0.3 — сильный.", "",
              "| признак | IV |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in iv.head(15).items()]

    # Гипотезы
    checks = []
    by_pay = train.groupby(train["PAY_1"].clip(-2, 3))[TARGET].mean()
    checks.append(("H1. Просрочка в последнем месяце — главный сигнал",
                   f"дефолт при PAY_1 ≤ 0: {train.loc[train.PAY_1 <= 0, TARGET].mean():.2f}, "
                   f"при PAY_1 = 1: {by_pay[1]:.2f}, при PAY_1 ≥ 2: {train.loc[train.PAY_1 >= 2, TARGET].mean():.2f}; "
                   f"IV(PAY_1) = {iv['PAY_1']:.2f} — максимум среди исходных признаков", "подтверждается"))
    r = rate_by_quantile(train, "util_mean")
    checks.append(("H2. Высокая утилизация лимита → выше риск",
                   f"дефолт по квинтилям util_mean: {' / '.join(f'{v:.2f}' for v in r)}", "подтверждается, но нелинейно"))
    r = rate_by_quantile(train, "pay_ratio_mean")
    checks.append(("H3. Кто гасит малую долю счёта, рискованнее",
                   f"дефолт по квинтилям pay_ratio_mean: {' / '.join(f'{v:.2f}' for v in r)}", "подтверждается"))
    r = rate_by_quantile(train, "LIMIT_BAL")
    checks.append(("H4. Больший лимит → ниже риск (банк уже отобрал клиента)",
                   f"дефолт по квинтилям LIMIT_BAL: {' / '.join(f'{v:.2f}' for v in r)}", "подтверждается"))
    sex = train.groupby("SEX")[TARGET].mean()
    age = rate_by_quantile(train, "AGE")
    checks.append(("H5. Демография — слабый сигнал",
                   f"мужчины {sex[1]:.2f} / женщины {sex[2]:.2f}; по квинтилям возраста "
                   f"{' / '.join(f'{v:.2f}' for v in age)}; IV(AGE) = {iv['AGE']:.3f}, IV(SEX) = {iv['SEX']:.3f}",
                   "подтверждается (и использовать пол в скоринге юридически спорно)"))
    corr = train[BILL_COLS].corr().to_numpy()
    off = corr[np.triu_indices(6, 1)]
    checks.append(("H6. Суммы счетов по месяцам мультиколлинеарны",
                   f"корреляции BILL_AMT1..6 между собой: {off.min():.2f}–{off.max():.2f}",
                   "подтверждается → для линейной модели лучше агрегаты (утилизация, тренд)"))
    lines += ["", "## Гипотезы", "", "| гипотеза | что видно в данных | вывод |", "|---|---|---|"]
    lines += [f"| {h} | {e} | {v} |" for h, e, v in checks]

    # Графики
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    counts = train["PAY_1"].clip(-2, 4).value_counts().sort_index()
    rates = train.groupby(train["PAY_1"].clip(-2, 4))[TARGET].mean()
    ax.bar(counts.index.astype(str), counts.values, color="#9db4d1")
    ax2 = ax.twinx()
    ax2.plot(rates.index.astype(str), rates.values, "o-", color="#c0392b")
    ax.set_title("PAY_1: число клиентов и доля дефолтов"); ax2.set_ylim(0, 1)
    for ax, col, title in [(axes[0, 1], "util_mean", "Утилизация лимита"),
                           (axes[0, 2], "pay_ratio_mean", "Доля погашения"),
                           (axes[1, 0], "LIMIT_BAL", "Кредитный лимит"),
                           (axes[1, 1], "AGE", "Возраст")]:
        r = rate_by_quantile(train, col, q=10)
        ax.bar(range(1, 11), r.values, color="#4a78b5")
        ax.axhline(base, ls="--", color="grey", lw=1)
        ax.set_title(f"{title}: дефолт по децилям"); ax.set_xticks(range(1, 11))
    ax = axes[1, 2]
    iv.head(12)[::-1].plot.barh(ax=ax, color="#4a78b5")
    ax.set_title("Information Value, топ-12")
    fig.tight_layout()
    fig.savefig(FIG / "eda.png", dpi=110)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(6), [f"B{i}" for i in range(1, 7)]); ax.set_yticks(range(6), [f"B{i}" for i in range(1, 7)])
    fig.colorbar(im); ax.set_title("Корреляции BILL_AMT1..6")
    fig.tight_layout(); fig.savefig(FIG / "bill_corr.png", dpi=110)

    (ROOT / "reports" / "eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
