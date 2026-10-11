import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import load_data
from models import ResNet
from train import train, evaluate


def check_indices(indices, pool_size):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("duplicate indices")
    if np.any(indices < 0) or np.any(indices >= pool_size):
        raise ValueError("index outside the training pool")


@torch.inference_mode()
def select(model, pool, remaining, count, device, batch_size=128, workers=0):
    if count < 1 or count > len(remaining):
        raise ValueError("selection exceeds the remaining pool")
    model.to(device).eval()
    amp = torch.device(device).type == "cuda"
    loader = DataLoader(Subset(pool, remaining.tolist()), batch_size, num_workers=workers)
    scores = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x.to(device))
        if amp and not torch.isfinite(logits).all():
            logits = model(x.to(device))
        confidence = logits.float().softmax(1).amax(1)
        if not torch.isfinite(confidence).all():
            raise ValueError("non-finite predictions")
        scores.extend((1 - confidence).cpu().tolist())
    order = np.argsort(-np.asarray(scores), kind="stable")[:count]
    return remaining[order]


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run(mode, training, pool, test, args, output):
    if mode not in ("random", "target", "proxy"):
        raise ValueError("unknown mode")
    if not training.train or not pool.train or test.train:
        raise ValueError("expected CIFAR train/train/test splits")
    if training is test or pool is test or not np.array_equal(training.data, pool.data):
        raise ValueError("training and selection must use the same training images")

    pool_size = min(512, len(pool)) if args.smoke else len(pool)
    if args.initial < 1 or not args.rounds or any(count < 1 for count in args.rounds):
        raise ValueError("initial and round sizes must be positive")
    if args.initial + sum(args.rounds) > pool_size:
        raise ValueError("labeling budget exceeds the training pool")

    rng = np.random.default_rng(args.seed)
    labeled = rng.permutation(pool_size)[:args.initial]
    check_indices(labeled, pool_size)
    sets = [labeled.tolist()]
    folder = output / f"{mode}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    test_size = min(384, len(test)) if args.smoke else len(test)
    config = {**vars(args), "mode": mode, "pool_size": pool_size,
        "test_size": test_size, "torch": torch.__version__}
    precision = "fp16" if torch.device(args.device).type == "cuda" else "fp32"
    config["precision"] = precision
    config["paper_commit"] = "348f90ac9caaa393117f1968f1aedf05dbfbac0f"
    if torch.device(args.device).type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(args.device)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    selection_seconds = 0

    for round_id, count in enumerate(args.rounds):
        sync(args.device)
        start = time.perf_counter()
        remaining = np.setdiff1d(np.arange(pool_size), labeled, assume_unique=True)
        if mode == "random":
            added = rng.choice(remaining, count, replace=False)
        else:
            torch.manual_seed(args.seed + round_id)
            model = ResNet(20 if mode == "proxy" else 164)
            loader = DataLoader(Subset(training, labeled.tolist()), args.batch_size,
                shuffle=True, num_workers=args.workers)
            history = train(model, loader, args.device, args.epochs, args.learning_rates)
            path = folder / f"round_{round_id + 1}.json"
            path.write_text(json.dumps(history, indent=2) + "\n")
            added = select(model, pool, remaining, count, args.device,
                args.batch_size, args.workers)
            del model
        sync(args.device)
        selection_seconds += time.perf_counter() - start

        check_indices(added, pool_size)
        if len(added) != count or np.intersect1d(labeled, added).size:
            raise ValueError("invalid selection")
        labeled = np.concatenate([labeled, added])
        check_indices(labeled, pool_size)
        if len(labeled) != args.initial + sum(args.rounds[:round_id + 1]):
            raise ValueError("incorrect labeled set size")
        sets.append(labeled.tolist())
        indices = {"split": "cifar10/train", "pool_size": pool_size, "sets": sets}
        (folder / "indices.json").write_text(json.dumps(indices) + "\n")
        print(f"{mode}: round {round_id + 1}, labeled={len(labeled)}")

    torch.manual_seed(args.seed + len(args.rounds))
    sync(args.device)
    start = time.perf_counter()
    model = ResNet(164)
    loader = DataLoader(Subset(training, labeled.tolist()), args.batch_size,
        shuffle=True, num_workers=args.workers)
    history = train(model, loader, args.device, args.epochs, args.learning_rates)
    sync(args.device)
    train_seconds = time.perf_counter() - start
    evaluation = Subset(test, range(min(384, len(test)))) if args.smoke else test
    test_loader = DataLoader(evaluation, args.batch_size, num_workers=args.workers)
    metrics = evaluate(model, test_loader, args.device)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite evaluation")
    result = {"mode": mode, "seed": args.seed, "precision": precision, "budget": len(labeled),
        "selection_seconds": selection_seconds, "train_seconds": train_seconds, **metrics}
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 active learning")
    parser.add_argument("--mode", choices=("random", "target", "proxy", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial", type=int, default=1000)
    parser.add_argument("--rounds", type=int, nargs="+", default=[4000, 5000, 5000, 5000, 5000])
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 90, 45, 45])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.01, .1, .01, .001])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.short and args.smoke:
        parser.error("choose short or smoke")
    if args.short:
        args.initial, args.rounds = 1000, [2000, 2000, 2500, 2500]
        args.epochs, args.learning_rates = [1, 6, 4, 4], [.01, .1, .01, .001]
    if args.smoke:
        args.initial, args.rounds = 128, [128, 128]
        args.epochs, args.learning_rates = [1], [.01]
    if args.seed < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("invalid seed, batch size or workers")
    if len(args.epochs) != len(args.learning_rates):
        parser.error("epochs and learning-rates must have the same length")
    bad_rates = any(lr <= 0 or not np.isfinite(lr) for lr in args.learning_rates)
    if any(n < 1 for n in args.epochs) or bad_rates:
        parser.error("epochs and learning-rates must be positive")

    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    training, pool, test = load_data()
    folder = "results/short" if args.short else "results/full"
    output = Path(args.output or ("results/modes_smoke" if args.smoke else folder))
    modes = ("random", "target", "proxy") if args.mode == "all" else (args.mode,)
    for mode in modes:
        result = run(mode, training, pool, test, args, output)
        path = output / "runs.csv"
        if path.exists():
            with path.open() as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row.setdefault("precision", "fp32")
        else:
            rows = []
        rows = [r for r in rows if (r["mode"], int(r["seed"])) != (mode, args.seed)]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows + [result])
        print(result)
