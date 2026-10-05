# Анализ данных: машинное обучение и BigData

Задания курса для заочного отделения, направление 09.03.02.
Преподаватель — Павлова Е.А. Язык — Python.

## Задания

| Задание | Ссылка |
| --- | --- |
| №1 — Сегментация клиентов методом K-средних | [assignment-1](https://github.com/evn1111/BigData/tree/assignment-1) |
| №2 — Моделирование производительности сотрудников | [assignment-2](https://github.com/evn1111/BigData/tree/assignment-2) |

Каждое задание выполняется в отдельной ветке, созданной от `main`. Код, данные и отчёт находятся в корне соответствующей ветки. Ветки заданий не объединяются в `main`.

Ветки `assignment-3` и `assignment-4` появятся по мере выполнения заданий.

## Запуск первого задания

```bash
python3 -m venv .venv
git switch assignment-1
.venv/bin/python -m pip install -r requirements.txt
MPLBACKEND=Agg .venv/bin/python customer_segmentation_kmeans.py
```

Данные, графики и отчёт находятся в папке задания. Учебные шаблоны и материалы курса в репозиторий не включены.
