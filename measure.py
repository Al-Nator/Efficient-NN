import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pynvml
import time
import threading

from models import CNN

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

model = CNN().cuda().eval()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def measure_latency(model, x, warmup=10, repeats=30):
    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        torch.cuda.synchronize()

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            model(x)
            end.record()

            torch.cuda.synchronize()

            times.append(start.elapsed_time(end) / 1000)

    return np.median(times)


def measure_memory(model, x):
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        model(x)

    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated()


def measure_energy(model, x, repeats=100):
    powers = []
    running = True

    def sample_power():
        while running:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000 
            powers.append(power)
            time.sleep(0.01)

    with torch.inference_mode():
        for _ in range(10):
            model(x)

        torch.cuda.synchronize()

        thread = threading.Thread(target=sample_power)
        thread.start()

        start = time.perf_counter()

        for _ in range(repeats):
            model(x)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        running = False
        thread.join()

    avg_power = np.mean(powers)

    return avg_power * elapsed / repeats


def measure_one(S, B):
    try:
        x = torch.randn(B, 3, S, S, device="cuda")

        latency = measure_latency(model, x)
        memory = measure_memory(model, x)
        energy = measure_energy(model, x)
        del x
        torch.cuda.empty_cache()

        return {"S": S, "B": B, "latency": latency, "memory": memory, "energy": energy, "oom": False}

    except torch.cuda.OutOfMemoryError:
        if "x" in locals():
            del x
        torch.cuda.empty_cache()

        return {"S": S, "B": B, "latency": None, "memory": None, "energy": None, "oom": True}



if __name__ == "__main__":
    rng = np.random.default_rng(42)

    base_sizes = [32, 64, 128, 224, 256, 384, 512]
    base_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    size_candidates = [s for s in range(32, 513, 16) if s not in base_sizes]
    random_sizes = rng.choice(size_candidates, size=4, replace=False).tolist()
    batch_candidates = [b for b in range(1, 257) if b not in base_batches and (b & (b - 1)) != 0]
    random_batches = rng.choice(batch_candidates, size=3, replace=False).tolist()

    sizes = base_sizes + random_sizes
    batches = base_batches + random_batches

    results = []

    for S in sizes:
        for B in batches:
            row = measure_one(S, B)
            row["is_validation"] = (S in random_sizes or B in random_batches)
            results.append(row)

            print(
                f"S={S:3d}, B={B:3d}, "
                f"latency={row['latency']}, "
                f"memory={row['memory']}, "
                f"energy={row['energy']}, "
                f"OOM={row['oom']}"
            )

    df = pd.DataFrame(results)
    df.to_csv("results/measurements.csv", index=False)

    print("random sizes:", random_sizes)
    print("random batches:", random_batches)
    print("saved:", len(df), "rows")