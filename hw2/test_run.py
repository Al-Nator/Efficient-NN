import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import run


class Images:
    def __init__(self, size=30, training=True):
        self.train = training
        self.data = np.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor([float(i)]), 0


class Scores(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x, torch.zeros_like(x)], dim=1)


class Checks(unittest.TestCase):
    def test_indices(self):
        for bad in ([1, 1], [-1], [30], [1.5], [[1]]):
            with self.assertRaises(ValueError):
                run.check_indices(bad, 30)
        run.check_indices(np.arange(30), 30)

    def test_least_confidence(self):
        remaining = np.array([3, 1, 2])
        selected = run.select(Scores(), Images(), remaining, 2, "cpu", batch_size=2)
        self.assertEqual(selected.tolist(), [1, 2])
        with self.assertRaises(ValueError):
            run.select(Scores(), Images(), remaining, 4, "cpu")

    def test_modes(self):
        training, pool, test = Images(), Images(), Images(10, False)
        args = SimpleNamespace(seed=0, smoke=False, initial=4, rounds=[3, 5],
            batch_size=2, workers=0, device="cpu", epochs=[1], learning_rates=[.01])
        created = []
        trained = []

        def model(depth):
            created.append(depth)
            return Scores()

        def train(model, loader, *args):
            self.assertIs(loader.dataset.dataset, training)
            run.check_indices(loader.dataset.indices, len(training))
            trained.append(loader.dataset.indices)
            return [{"loss": 1.0}]

        def evaluate(model, loader, device):
            self.assertIs(loader.dataset, test)
            return {"loss": 1.0, "accuracy": 10.0, "error": 90.0}

        expected = {"random": [164], "target": [164, 164, 164], "proxy": [20, 20, 164]}
        initial = []
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(run, ResNet=model, train=train, evaluate=evaluate):
                for mode in expected:
                    created.clear()
                    trained.clear()
                    result = run.run(mode, training, pool, test, args, Path(directory))
                    path = Path(directory) / f"{mode}_seed0" / "indices.json"
                    sets = json.loads(path.read_text())["sets"]
                    self.assertEqual([len(s) for s in sets], [4, 7, 12])
                    for previous, current in zip(sets, sets[1:]):
                        self.assertTrue(set(previous) < set(current))
                        run.check_indices(current, 30)
                    initial.append(sets[0])
                    self.assertEqual(created, expected[mode])
                    self.assertEqual(trained, [sets[-1]] if mode == "random" else sets)
                    self.assertEqual(result["budget"], 12)
                self.assertEqual(initial, [initial[0]] * 3)
                with self.assertRaises(ValueError):
                    run.run("proxy", training, pool, pool, args, Path(directory))
                args.rounds = [30]
                with self.assertRaises(ValueError):
                    run.run("random", training, pool, test, args, Path(directory))

    def test_full_budget(self):
        training, pool, test = Images(50000), Images(50000), Images(10000, False)
        args = SimpleNamespace(seed=0, smoke=False, initial=1000,
            rounds=[4000, 5000, 5000, 5000, 5000], batch_size=128,
            workers=0, device="cpu", epochs=[1], learning_rates=[.01])
        metrics = {"loss": 1., "accuracy": 10., "error": 90.}
        replacements = {"ResNet": lambda _: Scores(), "train": lambda *a: [],
            "evaluate": lambda *a: metrics, "select": lambda m, p, ids, n, *a: ids[:n]}
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(run, **replacements):
                for mode in ("random", "target", "proxy"):
                    run.run(mode, training, pool, test, args, Path(directory))
                    path = Path(directory) / f"{mode}_seed0" / "indices.json"
                    sets = json.loads(path.read_text())["sets"]
                    self.assertEqual([len(s) for s in sets], [1000, 5000, 10000, 15000, 20000, 25000])
                    for previous, current in zip(sets, sets[1:]):
                        run.check_indices(current, 50000)
                        self.assertTrue(set(previous) < set(current))


if __name__ == "__main__":
    unittest.main()
