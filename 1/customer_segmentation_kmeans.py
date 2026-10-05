# Задание 1. Сегментация клиентов

from pathlib import Path

import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 42
N_CUSTOMERS = 600
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def generate_customer_data(n_customers: int = N_CUSTOMERS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    segment_sizes = [180, 150, 150, n_customers - 480]
    # Генерируем клиентов с разными привычками покупок.
    profiles = [
        (30, 45, 3.0),    # молодые клиенты с редкими недорогими покупками
        (42, 110, 6.0),   # регулярные клиенты со средним чеком
        (48, 260, 3.5),   # клиенты с высоким средним чеком
        (55, 180, 12.0),  # самые активные постоянные клиенты
    ]

    rows = []
    for size, (age_mean, value_mean, frequency_mean) in zip(segment_sizes, profiles):
        rows.append(
            pd.DataFrame(
                {
                    "Age": np.clip(rng.normal(age_mean, 7, size), 18, 75).round().astype(int),
                    "AveragePurchaseValue": np.clip(
                        rng.normal(value_mean, value_mean * 0.22, size), 10, None
                    ).round(2),
                    "PurchaseFrequency": np.clip(
                        rng.poisson(frequency_mean, size) + 1, 1, 30
                    ).astype(int),
                }
            )
        )

    customers = pd.concat(rows, ignore_index=True)
    customers = customers.sample(frac=1, random_state=SEED).reset_index(drop=True)
    customers.insert(0, "CustomerID", np.arange(100001, 100001 + len(customers)))
    return customers


def save_eda(data: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    data.to_csv(DATA_DIR / "customers.csv", index=False)
    data.describe().T.to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, column, title in zip(
        axes,
        ["Age", "AveragePurchaseValue", "PurchaseFrequency"],
        ["Age distribution", "Average purchase value", "Purchase frequency"],
    ):
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_feature_distributions.png", dpi=160)
    plt.close(fig)


def select_k(features: np.ndarray) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = model.fit_predict(features)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(features, labels),
            }
        )
    metrics = pd.DataFrame(rows)
    # Выбираем k с самым высоким коэффициентом силуэта.
    best_k = int(metrics.loc[metrics["silhouette_score"].idxmax(), "k"])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(metrics["k"], metrics["inertia"], marker="o", color="#2563eb")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(metrics["k"], metrics["silhouette_score"], marker="s", color="#dc2626")
    ax2.set_ylabel("Silhouette score", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax1.axvline(best_k, linestyle="--", color="#111827", alpha=0.7, label=f"Selected k={best_k}")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_k_selection.png", dpi=160)
    plt.close(fig)
    return best_k, metrics


def cluster_customers(data: pd.DataFrame, best_k: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    feature_columns = ["Age", "AveragePurchaseValue", "PurchaseFrequency"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])
    model = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    labels = model.fit_predict(scaled_features)

    result = data.copy()
    result["Cluster"] = labels + 1
    profiles = (
        result.groupby("Cluster")[feature_columns]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    profiles.to_csv(RESULTS_DIR / "cluster_profiles.csv")
    result.to_csv(RESULTS_DIR / "customers_with_clusters.csv", index=False)

    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(scaled_features)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=labels + 1, cmap="viridis", alpha=0.7, s=28
    )
    ax.set_title("Customer clusters after PCA projection")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_customer_clusters.png", dpi=160)
    plt.close(fig)

    score = silhouette_score(scaled_features, labels)
    return result, profiles, score


def write_report(data: pd.DataFrame, metrics: pd.DataFrame, best_k: int, score: float) -> None:
    cluster_stats = (
        data.groupby("Cluster")[["Age", "AveragePurchaseValue", "PurchaseFrequency"]]
        .mean()
        .round(1)
    )
    recommendations = {
        "low": "предлагать недорогие товары, welcome-скидки и рекомендации для увеличения частоты покупок",
        "regular": "использовать персональные подборки, бонусную программу и скидки за повторную покупку",
        "premium": "предлагать более дорогие товары и ранний доступ к новым коллекциям",
        "active": "предлагать сопутствующие товары и накопительные бонусы за покупки",
    }
    ordered = cluster_stats.sort_values("AveragePurchaseValue")
    rows = []
    for cluster, row in ordered.iterrows():
        if row["PurchaseFrequency"] == ordered["PurchaseFrequency"].max():
            profile = "active"
        elif row["AveragePurchaseValue"] == ordered["AveragePurchaseValue"].max():
            profile = "premium"
        elif row["PurchaseFrequency"] < ordered["PurchaseFrequency"].median():
            profile = "low"
        else:
            profile = "regular"
        rows.append(
            f"- Кластер {cluster}: {int((data['Cluster'] == cluster).sum())} клиентов; "
            f"средний возраст {row['Age']:.1f}; средний чек ${row['AveragePurchaseValue']:.2f}; "
            f"частота покупок {row['PurchaseFrequency']:.1f}. Рекомендация: {recommendations[profile]}."
        )

    report = f"""# Задание №1. Сегментация клиентов методом K-средних

## Цель

Разделить клиентов интернет-магазина на группы и подобрать для каждой группы подходящие предложения.

## Данные

Для задания с помощью Python создан набор из {len(data)} клиентов. Для каждого клиента указаны возраст, средняя сумма покупки в долларах и число покупок за полгода. CustomerID — это номер клиента, поэтому в расчёт кластеров он не включён. Данные искусственные: они подходят для учебного примера, но не описывают реальный магазин.

В таблице {len(data.columns) - 1} исходных столбца. Пропусков: {int(data.drop(columns='Cluster').isna().sum().sum())}. Повторяющихся номеров клиентов: {int(data['CustomerID'].duplicated().sum())}.

Возраст клиентов: от {data['Age'].min()} до {data['Age'].max()} лет. Средний чек: от {data['AveragePurchaseValue'].min():.2f} до {data['AveragePurchaseValue'].max():.2f} долларов. Число покупок: от {data['PurchaseFrequency'].min()} до {data['PurchaseFrequency'].max()} за полгода. По графикам видно, что небольшие чеки и редкие покупки встречаются чаще, чем большие чеки и частые покупки.

## Методика

Возраст, сумма покупки и частота покупок имеют разные масштабы. Поэтому перед расчётом использован StandardScaler: из каждого значения вычитается среднее и результат делится на стандартное отклонение. Так сумма покупки не будет влиять на расстояния сильнее остальных признаков только из-за своих больших значений.

Проверены варианты от 2 до 8 кластеров. Для каждого построена модель K-средних и рассчитан коэффициент силуэта. Он показывает, насколько клиенты похожи на свою группу и отличаются от соседних групп. Лучший результат получен при k = **{best_k}**. На графике локтя после этого значения снижение инерции становится менее резким. Для повторения результата используется random_state={SEED}.

Для графика кластеров три признака сведены к двум с помощью PCA. Это нужно только для изображения: сама кластеризация выполнена по всем трём признакам.

## Результаты

Коэффициент силуэта итоговой модели: **{score:.3f}**.

{{clusters}}

## Вывод

Получено {best_k} группы клиентов. Они отличаются средним чеком и числом покупок. Эти различия можно использовать при подготовке рассылок: постоянным клиентам предлагать бонусы, покупателям с большим чеком — более дорогие товары, а остальным — скидки на повторную покупку.

Коэффициент силуэта {score:.3f} показывает, что группы выделяются, но частично пересекаются. Рекомендации пока являются предположениями: проверить их эффективность можно только на реальных данных и результатах рекламных кампаний.

Подробные данные сохранены в `data/customers.csv` и `results/customers_with_clusters.csv`, графики — в `results/figures/`.
""".replace("{clusters}", "\n".join(rows))
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    metrics.to_csv(RESULTS_DIR / "k_selection_metrics.csv", index=False)


def main() -> None:
    data = generate_customer_data()
    save_eda(data)
    feature_columns = ["Age", "AveragePurchaseValue", "PurchaseFrequency"]
    scaled = StandardScaler().fit_transform(data[feature_columns])
    best_k, metrics = select_k(scaled)
    clustered, _, score = cluster_customers(data, best_k)
    write_report(clustered, metrics, best_k, score)
    print(f"Готово: {len(clustered)} клиентов, выбрано кластеров: {best_k}, silhouette: {score:.3f}")
    print(f"Результаты: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
