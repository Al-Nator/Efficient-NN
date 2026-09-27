# Homework 1 — Analytical Performance Model of a Small CNN

## Overview

This project builds and validates analytical models for the cost of one FP32 forward pass of a small convolutional neural network.

The following quantities are modeled as functions of image size `S` and batch size `B`:

- `FLOPs(S, B)` — number of floating-point operations;
- `Memory(S, B)` — estimated GPU memory usage;
- `Latency(S, B, θ)` — forward-pass latency;
- `Energy(S, B, θ)` — energy consumed by one forward pass.

The analytical predictions are compared with measurements on a real GPU. Latency and energy model parameters are calibrated only on the base grid, while randomly sampled image and batch sizes are used as validation points.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB
- Python: 3.12.3
- PyTorch: CUDA 12.8 build
- NumPy: 2.5.3
- pandas: 3.0.6
- SciPy: 1.18.1
- Matplotlib: 3.11.2
- nvidia-ml-py: 13.615.71
- Environment manager: `uv`

All measurements use FP32 with TF32 disabled:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
```

The model is evaluated with `model.eval()` and `torch.inference_mode()`.

## Repository structure

```text
hw1/
├── README.md
├── hw1_handwritten.pdf
├── models.py
├── equations.py
├── measure.py
├── calibrate.py
└── results/
    ├── measurements.csv
    ├── theta.json
    └── figures/
```

## Analytical model

### FLOPs

One multiply-accumulate is counted as two FLOPs.

For a convolution:

```text
FLOPs = 2 * B * Cout * Hout * Wout * Cin * K²
```

For the complete network:

```text
FLOPs(S, B) = B * (17712 * S² + 313344)
```

ReLU, pooling and global-average-pooling FLOPs are ignored. Linear bias additions are also ignored.

### Memory

The analytical memory model uses a deliberately simple conservative approximation:

```text
Memory = parameters + input + sum of all activations
```

All tensors are FP32, so every stored value occupies 4 bytes.

The resulting model is:

```text
Memory(S, B) = 104 * B * S² + 3472 * B + 4161296 bytes
```

ReLU is in-place and therefore does not introduce an additional activation buffer.

### Bytes moved

Memory traffic is modeled in a worst-case manner without assuming cache reuse.

For each layer, input reads, parameter reads and output writes are counted. ReLU, MaxPool and GlobalAvgPool memory traffic is also included.

The resulting expression is:

```text
BytesMoved(S, B) = 364 * B * S² + 8592 * B + 4161296 bytes
```

### Latency

Latency is modeled as a fixed launch overhead plus the slower of compute and memory-transfer times:

```text
Latency = launch + max(
    FLOPs / effective_compute_rate,
    BytesMoved / effective_memory_bandwidth
)
```

Calibrated effective parameters:

```text
launch                  ≈ 47.3 µs
effective compute       ≈ 14.2 TFLOP/s
effective bandwidth     ≈ 296 GB/s
```

These values are fitted effective parameters and should not be interpreted as the nominal peak specifications of the GPU.

### Energy

Energy is modeled as the sum of baseline, arithmetic and memory-transfer contributions:

```text
Energy =
    base_power * Latency
    + joule_per_gflop * FLOPs / 1e9
    + joule_per_gbyte * BytesMoved / 1e9
```

Fitted parameters:

```text
base_power        ≈ 1.34e-10 W
joule_per_gflop   ≈ 0.01993 J/GFLOP
joule_per_gbyte   ≈ 5.20e-14 J/GB
```

For this fixed CNN, FLOPs, memory traffic and latency are strongly correlated. Therefore the individual energy coefficients are poorly identifiable: the fitted model assigns almost all variation to the FLOPs term. These coefficients should be treated as empirical fitting parameters rather than physical GPU characteristics.

## Measurement protocol

Base image sizes:

```text
32, 64, 128, 224, 256, 384, 512
```

Additional validation image sizes sampled with seed 42:

```text
80, 272, 352, 400
```

Base batch sizes:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Additional validation batch sizes:

```text
29, 56, 179
```

This gives `11 × 12 = 132` configurations.

Latency is measured with CUDA events using 10 warm-up iterations and the median of 30 forward passes.

Peak memory is measured with:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Energy is estimated from whole-GPU NVML power measurements. Because a single forward pass can be much shorter than the power-sampling interval, 100 consecutive forward passes are measured and the total energy is divided by 100.

No OOM configuration was observed on the RTX 4070 Ti SUPER for the required measurement grid.

## Results and discussion

The calibrated latency model matches both calibration and validation points closely. At small batch sizes, latency changes very little as `B` increases, which is consistent with a launch-bound regime where fixed GPU overhead dominates. As the batch size increases, latency grows approximately linearly and computation/memory traffic become dominant.

The analytical memory model systematically overestimates the measured peak for most large configurations. This is expected because the analytical model sums all activations, while the real inference execution can release or reuse intermediate buffers. Some measured points also show discontinuities caused by different CUDA/cuDNN algorithms and temporary workspaces selected for particular tensor shapes.

The energy model gives a very accurate empirical fit over the tested grid and generalizes well to the held-out validation points. However, this should not be interpreted as an accurate decomposition of physical GPU energy consumption. For a fixed architecture, FLOPs, memory traffic and latency scale similarly with `B` and `S`, making their individual energy contributions difficult to identify independently.

The `latency_vs_batch.png` plot illustrates the transition most clearly: small batches are dominated by nearly constant overhead, while larger batches move toward memory/compute-bound behavior.

## Reproduction

Install the locked environment:

```bash
uv sync
```

Run measurements:

```bash
uv run measure.py
```

Calibrate the analytical models and regenerate figures:

```bash
uv run calibrate.py
```

Outputs are written to:

```text
results/measurements.csv
results/theta.json
results/figures/
```