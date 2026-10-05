# Задание №1 — сегментация клиентов методом K-средних

Установка зависимостей из корня проекта:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Запуск из корня проекта:

```bash
MPLCONFIGDIR=/tmp/analyzedata-mpl .venv/bin/python 1/customer_segmentation_kmeans.py
```

Скрипт создаёт:

- `data/customers.csv` — сгенерированный набор клиентов;
- `results/descriptive_statistics.csv` — описательную статистику;
- `results/k_selection_metrics.csv` — метрики для выбора числа кластеров;
- `results/customers_with_clusters.csv` — клиенты с присвоенными кластерами;
- `results/cluster_profiles.csv` — профили кластеров;
- `results/report.md` — краткий текстовый анализ и рекомендации;
- `results/figures/` — графики распределений, выбора `k` и кластеров.
