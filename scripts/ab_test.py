"""
Симуляция A/B-теста: старая политика (логистическая регрессия) против новой (лучшая модель).

На отложенном test (6 000 заявителей) много раз случайно делим клиентов на группы A и B,
в каждой группе своя модель одобряет 70% заявок, сравниваем bad rate среди одобренных.
Плюс A/A-тест (обе группы — старая модель): доля p < 0.05 должна быть около 5%.

    python scripts/ab_test.py     # после scripts/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.ab import run_policy_test, sample_size_per_arm  # noqa: E402

BASELINE = "logreg (сырые признаки) — бейзлайн"
APPROVAL = 0.7
N_SIM = 2000


def main():
    preds = pd.read_csv(ROOT / "data" / "test_predictions.csv")
    best = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))["best_model"]
    y = preds["y"].to_numpy()
    rng = np.random.default_rng(0)

    ab = pd.DataFrame([run_policy_test(y, preds[BASELINE], preds[best], APPROVAL, rng) for _ in range(N_SIM)])
    aa = pd.DataFrame([run_policy_test(y, preds[BASELINE], preds[BASELINE], APPROVAL, rng) for _ in range(N_SIM)])

    # Истинный эффект: обе политики на всех 6 000 заявителях
    k = int(APPROVAL * len(y))
    rate_a = y[np.argsort(preds[BASELINE].to_numpy(), kind="stable")[:k]].mean()
    rate_b = y[np.argsort(preds[best].to_numpy(), kind="stable")[:k]].mean()
    n_needed = sample_size_per_arm(rate_a, rate_b)
    one = ab.iloc[0]

    md = [
        "# A/B-тест кредитной политики (симуляция на test)", "",
        f"Группа A — бейзлайн (логистическая регрессия на сырых признаках), группа B — {best}. "
        f"В каждой группе одобряется {APPROVAL:.0%} заявок с наименьшим PD. Метрика — доля дефолтов среди одобренных "
        "(bad rate). Гипотеза H0: bad rate одинаков. Тест — двусторонний z-тест для двух долей, α = 0.05.", "",
        "## Один эксперимент", "",
        f"A: {int(one.bad_a)} дефолтов из {int(one.n_a)} одобренных ({one.rate_a:.3f}); "
        f"B: {int(one.bad_b)} из {int(one.n_b)} ({one.rate_b:.3f}); z = {one.z:.2f}, p = {one.p_value:.4f}. "
        "Один тест такого размера легко не показывает эффект — см. мощность ниже.", "",
        f"## {N_SIM} случайных разбиений", "",
        "| | A/B (старая vs новая) | A/A (старая vs старая) |", "|---|---|---|",
        f"| средний bad rate A | {ab.rate_a.mean():.4f} | {aa.rate_a.mean():.4f} |",
        f"| средний bad rate B | {ab.rate_b.mean():.4f} | {aa.rate_b.mean():.4f} |",
        f"| доля экспериментов с p < 0.05 | **{(ab.p_value < 0.05).mean():.1%}** (мощность) | "
        f"**{(aa.p_value < 0.05).mean():.1%}** (ошибка I рода) |", "",
        "## Размер выборки", "",
        f"Истинный эффект на всём test: bad rate {rate_a:.4f} → {rate_b:.4f} "
        f"(−{(rate_a - rate_b) / rate_a:.0%} относительно). Для мощности 80% нужно ≈ **{n_needed:,}** одобренных "
        f"клиентов в каждой группе, у нас ≈ {int(ab.n_a.mean()):,}. Отсюда мощность меньше 80%.", "",
        "## Что важно в реальном тесте", "",
        "- Метка дефолта созревает месяцами: в реальном тесте bad rate измеряют через 3–12 месяцев после выдачи.",
        "- Отказанных клиентов не видно (reject inference): сравнивать модели на исторических одобренных — смещённо, "
        "A/B с рандомизацией это смещение убирает.",
        "- Guardrail-метрики: доля одобрений, средний лимит, прибыль, доля одобрений по сегментам (без дискриминации).",
        "- Рандомизировать надо по клиенту, а не по заявке: повторные заявки одного человека не должны попасть в разные группы.",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ab.rate_a, bins=40, alpha=0.6, label="A: бейзлайн")
    axes[0].hist(ab.rate_b, bins=40, alpha=0.6, label=f"B: {best}")
    axes[0].set(title="Bad rate среди одобренных по 2000 разбиениям"); axes[0].legend()
    axes[1].hist(aa.p_value, bins=20, alpha=0.6, label="A/A", density=True)
    axes[1].hist(ab.p_value, bins=20, alpha=0.6, label="A/B", density=True)
    axes[1].axvline(0.05, color="black", ls="--", lw=1)
    axes[1].set(title="Распределение p-value: A/A равномерное, A/B прижато к нулю"); axes[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / "ab_test.png", dpi=110)

    (ROOT / "reports" / "ab_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
