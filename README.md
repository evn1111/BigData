# Задание №3 — A/B-тестирование нового дизайна сайта

Задания курса для заочного отделения, направление 09.03.02.
Преподаватель — Павлова Е.А. Язык — Python.

## Запуск задания


```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLBACKEND=Agg .venv/bin/python ab_test.py
```

Код — `ab_test.py`, данные — `data/users.csv`, отчёт — `results/report.md`. Графики находятся в `results/figures/`, таблицы расчётов — в `results/`.

В симуляции участвуют 2400 пользователей. Выполнены 10 000 бутстрап-повторений и тест хи-квадрат. Текст отчёта хранится отдельно от кода.
