# ДЗ2. Selection via Proxy

Python 3.12, uv, NVIDIA GPU с драйвером для CUDA 12.8.
PyTorch 2.11.0, torchvision 0.26.0. Зависимости зафиксированы в uv.lock.

Состав:

- data.py - CIFAR-10 и преобразования изображений.
- models.py - PreAct ResNet-20 и ResNet-164.
- train.py - обучение и оценка модели.
- run.py - random, target и proxy с least confidence.
- plot.py - сводная статистика и графики.
- paper.pdf - статья.
- report.md, report.pdf - отчёт.
- slides.pptx, slides.pdf - презентация.
- results/short/ - результаты девяти запусков.
- assets/ - рисунки статьи для презентации.

Команды запускать из hw2/. CIFAR-10 скачивается автоматически в data/.

Установка окружения:

```bash
uv sync --locked
```

Эксперимент с бюджетом 10 000, 15 эпохами и тремя seed:

```bash
uv run run.py --short --mode all --seed 0 --device cuda
uv run run.py --short --mode all --seed 1 --device cuda
uv run run.py --short --mode all --seed 2 --device cuda
uv run plot.py
```

--short: пул 50 000, наборы 1 000 / 3 000 / 5 000 / 7 500 / 10 000, тест 10 000.
Результаты и графики сохраняются в results/short/.
Для отдельного режима используется --mode random, --mode target или --mode proxy.
Для другой папки результатов используется --output PATH.

Запуск с бюджетом 25 000 и 181 эпохой:

```bash
uv run run.py --mode all --seed 0 --device cuda
```

Результаты сохраняются в results/full/.
