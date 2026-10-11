# ДЗ1. Модель производительности CNN

Python 3.12, uv, NVIDIA GPU с драйвером для CUDA 12.8.
Зависимости указаны в pyproject.toml и зафиксированы в uv.lock.

Состав:

- models.py - архитектура CNN.
- equations.py - расчёт FLOPs, памяти, времени и энергии.
- measure.py - измерения на GPU.
- calibrate.py - подбор параметров и построение графиков.
- hw1_handwritten.pdf - отчёт.
- results/ - измерения, параметры и графики.

Установка и запуск из hw1/:

```bash
uv sync --locked
uv run measure.py
uv run calibrate.py
```

measure.py сохраняет results/measurements.csv.
calibrate.py сохраняет results/theta.json и графики в results/figures/.
