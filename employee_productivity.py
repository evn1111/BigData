from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


folder = Path(__file__).resolve().parent
data_folder = folder / "data"
results = folder / "results"
figures = results / "figures"
data_folder.mkdir(exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 600
age = rng.integers(18, 66, n)
experience = np.array([rng.integers(0, min(30, a - 18) + 1) for a in age])
hours = rng.integers(20, 61, n)
education = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.05, 0.25, 0.45, 0.20, 0.05])
productivity = (
    20 + 1.2 * hours + 1.8 * experience + 5 * education
    - 0.15 * age + rng.normal(0, 8, n)
)
data = pd.DataFrame({
    "Часы_работы": hours,
    "Опыт": experience,
    "Возраст": age,
    "Образование": education,
    "Производительность": productivity.round(2),
})
data.to_csv(data_folder / "employees.csv", index=False)
data.describe().T.to_csv(results / "statistics.csv")
checks = pd.DataFrame({
    "Тип": data.dtypes.astype(str),
    "Пропуски": data.isna().sum(),
    "Минимум": data.min(),
    "Максимум": data.max(),
})
checks.to_csv(results / "data_checks.csv")
assert data.isna().sum().sum() == 0
assert data["Часы_работы"].between(20, 60).all()
assert data["Опыт"].between(0, 30).all()
assert data["Возраст"].between(18, 65).all()
assert data["Образование"].between(0, 4).all()
assert (data["Опыт"] <= data["Возраст"] - 18).all()
print("Размер таблицы:", data.shape)
print("Повторяющиеся строки:", data.duplicated().sum())
print(checks)

X = data.drop(columns="Производительность")
y = data["Производительность"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear = LinearRegression()
linear.fit(X_train, y_train)
forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42)
forest.fit(X_train, y_train)

predictions = pd.DataFrame({"Факт": y_test})
metrics = []
for name, model in [("Линейная регрессия", linear), ("Случайный лес", forest)]:
    prediction = model.predict(X_test)
    predictions[name] = prediction
    metrics.append({
        "Модель": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "R2": r2_score(y_test, prediction),
    })
metrics = pd.DataFrame(metrics)
metrics.to_csv(results / "metrics.csv", index=False)
predictions["Ошибка_линейной"] = predictions["Факт"] - predictions["Линейная регрессия"]
predictions.to_csv(results / "predictions.csv", index_label="Строка")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
scaled_linear = LinearRegression().fit(X_scaled, y_train)
coefficients = pd.DataFrame({
    "Признак": X.columns,
    "Коэффициент": linear.coef_,
    "Коэффициент_после_стандартизации": scaled_linear.coef_,
})
coefficients.to_csv(results / "coefficients.csv", index=False)
print("\nМетрики на тестовой выборке:")
print(metrics.round(3).to_string(index=False))
print("\nКоэффициенты:")
print(coefficients.round(3).to_string(index=False))
print("Свободный член:", round(linear.intercept_, 3))
print("Средняя ошибка:", round(predictions["Ошибка_линейной"].mean(), 3))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, column in zip(axes.flat, X.columns):
    ax.scatter(data[column], y, alpha=0.35, s=12)
    ax.set_xlabel(column)
    ax.set_ylabel("Производительность")
fig.tight_layout()
fig.savefig(figures / "01_dependencies.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, ["Линейная регрессия", "Случайный лес"]):
    ax.scatter(y_test, predictions[name], alpha=0.6, s=20)
    limits = [min(y_test.min(), predictions[name].min()), max(y_test.max(), predictions[name].max())]
    ax.plot(limits, limits, "r--")
    ax.set_title(name)
    ax.set_xlabel("Фактическое значение")
    ax.set_ylabel("Прогноз")
fig.tight_layout()
fig.savefig(figures / "02_predictions.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(predictions["Линейная регрессия"], predictions["Ошибка_линейной"], alpha=0.6)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Прогноз")
axes[0].set_ylabel("Факт минус прогноз")
axes[1].hist(predictions["Ошибка_линейной"], bins=20, edgecolor="white")
axes[1].set_xlabel("Ошибка")
axes[1].set_ylabel("Число сотрудников")
fig.tight_layout()
fig.savefig(figures / "03_errors.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(X.columns, scaled_linear.coef_)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Изменение производительности при увеличении признака на одно стандартное отклонение")
fig.tight_layout()
fig.savefig(figures / "04_coefficients.png", dpi=150)
plt.close(fig)
