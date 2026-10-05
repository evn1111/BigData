# Задание №2 — моделирование производительности сотрудников

Задания курса для заочного отделения, направление 09.03.02.
Преподаватель — Павлова Е.А. Язык — Python.

## Запуск

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLBACKEND=Agg .venv/bin/python employee_productivity.py
```

Код находится в `employee_productivity.py`, данные — в `data/employees.csv`, отчёт — в `results/report.md`, графики — в `results/figures/`.

В работе сравниваются линейная регрессия и случайный лес. Метрики рассчитаны на 120 тестовых записях, которые не использовались для обучения.
