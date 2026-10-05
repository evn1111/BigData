from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


folder = Path(__file__).resolve().parent
data_folder = folder / "data"
results = folder / "results"
figures = results / "figures"
data_folder.mkdir(exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 1200
a = rng.binomial(1, 0.50, n)
b = rng.binomial(1, 0.55, n)
start = pd.Timestamp("2023-01-01")
end = pd.Timestamp("2024-01-01")
seconds = rng.integers(0, int((end - start).total_seconds()), 2 * n)
data = pd.DataFrame({
    "UserID": np.arange(1, 2 * n + 1),
    "Group": ["A"] * n + ["B"] * n,
    "Conversion": np.concatenate([a, b]),
    "Timestamp": start + pd.to_timedelta(seconds, unit="s"),
})
data["Month"] = data["Timestamp"].dt.month
data["Year"] = data["Timestamp"].dt.year
data.to_csv(data_folder / "users.csv", index=False)
assert data["UserID"].is_unique
assert data["Conversion"].isin([0, 1]).all()
assert data["Year"].eq(2023).all()
assert not data.isna().any().any()

difference = b.mean() - a.mean()
bootstrap_rng = np.random.default_rng(43)
bootstrap = np.empty(10000)
for i in range(len(bootstrap)):
    sample_a = bootstrap_rng.choice(a, size=n, replace=True)
    sample_b = bootstrap_rng.choice(b, size=n, replace=True)
    bootstrap[i] = sample_b.mean() - sample_a.mean()
lower, upper = np.percentile(bootstrap, [2.5, 97.5])
pd.DataFrame({"Difference_B_minus_A": bootstrap}).to_csv(
    results / "bootstrap.csv", index=False
)

table = pd.crosstab(data["Group"], data["Conversion"]).reindex(columns=[0, 1])
table.to_csv(results / "contingency_table.csv")
chi2, p_value, dof, expected = chi2_contingency(table, correction=False)
pd.DataFrame(expected, index=table.index, columns=table.columns).to_csv(
    results / "expected_counts.csv"
)
summary = pd.DataFrame([{
    "Conversion_A": a.mean(),
    "Conversion_B": b.mean(),
    "Difference_B_minus_A": difference,
    "Relative_lift": difference / a.mean(),
    "CI95_lower": lower,
    "CI95_upper": upper,
    "Chi_square": chi2,
    "Degrees_of_freedom": dof,
    "P_value": p_value,
}])
summary.to_csv(results / "test_results.csv", index=False)
monthly = data.groupby(["Month", "Group"])["Conversion"].agg(["count", "sum", "mean"])
monthly.to_csv(results / "monthly_conversion.csv")
yearly = data.groupby(["Year", "Group"])["Conversion"].agg(["count", "sum", "mean"])
yearly.to_csv(results / "yearly_conversion.csv")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(bootstrap * 100, bins=40, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", label="Нет различий")
ax.axvline(lower * 100, color="red", linestyle="--", label="95% доверительный интервал")
ax.axvline(upper * 100, color="red", linestyle="--")
ax.set_xlabel("Разница конверсий B − A, процентные пункты")
ax.set_ylabel("Число повторений")
ax.legend()
fig.tight_layout()
fig.savefig(figures / "01_bootstrap.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
for group in ["A", "B"]:
    values = monthly.xs(group, level="Group")["mean"]
    ax.plot(values.index, values * 100, marker="o", label=group)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Месяц 2023 года")
ax.set_ylabel("Конверсия, %")
ax.legend(title="Группа")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(figures / "02_monthly_conversion.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["A", "B"], [a.mean() * 100, b.mean() * 100], color=["steelblue", "orange"])
ax.bar_label(bars, fmt="%.2f%%")
ax.set_ylim(0, 100)
ax.set_ylabel("Конверсия за 2023 год, %")
fig.tight_layout()
fig.savefig(figures / "03_yearly_conversion.png", dpi=150)
plt.close(fig)

print(table)
print(summary.T.round(6))
print("Минимальная ожидаемая частота:", expected.min())
print("Различия значимы при уровне 0.05:", p_value < 0.05)
