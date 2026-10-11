import time

import torch
from torch.nn import functional as F


def train(model, loader, device, epochs=(1, 90, 45, 45),
    learning_rates=(0.01, 0.1, 0.01, 0.001), momentum=0.9, weight_decay=5e-4):
    model.to(device)
    amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=amp)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[0],
        momentum=momentum, weight_decay=weight_decay)
    history = []
    for count, lr in zip(epochs, learning_rates, strict=True):
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(count):
            start = time.perf_counter()
            model.train()
            loss_sum = correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            metrics = {"loss": loss_sum / total, "accuracy": 100 * correct / total}
            metrics["seconds"] = time.perf_counter() - start
            history.append({"epoch": len(history) + 1, "lr": lr, **metrics})
            message = f"epoch {len(history)}: loss={metrics['loss']:.4f}"
            print(f"{message}, acc={metrics['accuracy']:.2f}%, {metrics['seconds']:.2f}s")
    return history


@torch.inference_mode()
def evaluate(model, loader, device):
    model.to(device).eval()
    amp = torch.device(device).type == "cuda"
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        if amp and not torch.isfinite(logits).all():
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y, reduction="sum")
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    accuracy = 100 * correct / total
    return {"loss": loss_sum / total, "accuracy": accuracy, "error": 100 - accuracy}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from torch.utils.data import DataLoader, Subset
    from data import load_data
    from models import ResNet

    parser = argparse.ArgumentParser(description="Quick CIFAR-10 training check")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("batches and batch-size must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    training, _, test = load_data()
    size = args.batches * args.batch_size
    if size > min(len(training), len(test)):
        parser.error("too many batches for the dataset")
    results = []
    for depth in (20, 164):
        torch.manual_seed(0)
        loader = DataLoader(Subset(training, range(size)), args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test, range(size)), args.batch_size)
        model = ResNet(depth)
        before = model.conv.weight.detach().clone()
        history = train(model, loader, args.device, epochs=(1,), learning_rates=(0.01,))
        metrics = evaluate(model, test_loader, args.device)
        assert torch.isfinite(torch.tensor([history[0]["loss"], metrics["loss"]])).all()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert not torch.equal(before, model.conv.weight.detach().cpu())
        result = {"model": depth, "device": args.device, "seed": 0,
            "batches": args.batches, "batch_size": args.batch_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train": history[0], "test": metrics}
        results.append(result)
        print(f"ResNet-{depth}: {metrics}")
    Path("results").mkdir(exist_ok=True)
    Path("results/smoke.json").write_text(json.dumps(results, indent=2) + "\n")
