import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from equations import flops, memory, bytes_moved, latency, energy

df = pd.read_csv("results/measurements.csv")
train = df[(~df["oom"]) & (~df["is_validation"])].copy()

S = train["S"].to_numpy()
B = train["B"].to_numpy()
y_latency = train["latency"].to_numpy()

def latency_residuals(x):
    theta = {"launch": x[0], "compute": x[1], "bandwidth": x[2]}
    pred = latency(S, B, theta)
    return pred - y_latency

latency_fit = least_squares(latency_residuals, x0=[1e-4, 10e12, 500e9], bounds=([0, 1e9, 1e9], [1, 1e15, 1e13]))
theta_latency = {"launch": latency_fit.x[0], "compute": latency_fit.x[1], "bandwidth": latency_fit.x[2]}

latency_pred = latency(S, B, theta_latency)
flops_values = flops(S, B) / 1e9
bytes_values = bytes_moved(S, B) / 1e9
y_energy = train["energy"].to_numpy()

def energy_residuals(x):
    pred = x[0] * latency_pred + x[1] * flops_values + x[2] * bytes_values
    return pred - y_energy

energy_fit = least_squares(energy_residuals, x0=[30.0, 0.01, 0.01], bounds=(0, np.inf))
theta_energy = {"base_power": energy_fit.x[0], "joule_per_gflop": energy_fit.x[1], "joule_per_gbyte": energy_fit.x[2], "latency": theta_latency}
theta = {"latency": theta_latency, "energy": theta_energy}

os.makedirs("results/figures", exist_ok=True)

with open("results/theta.json", "w") as f:
    json.dump(theta, f, indent=2)

eval_df = df[~df["oom"]].copy()
S_all = eval_df["S"].to_numpy()
B_all = eval_df["B"].to_numpy()

eval_df["pred_memory"] = memory(S_all, B_all)
eval_df["pred_latency"] = latency(S_all, B_all, theta_latency)
eval_df["pred_energy"] = energy(S_all, B_all, theta_energy)

def plot_predicted_vs_measured(measured_col, predicted_col, xlabel, ylabel, title, filename):
    train_df = eval_df[~eval_df["is_validation"]]
    val_df = eval_df[eval_df["is_validation"]]

    x_train = train_df[measured_col].to_numpy()
    y_train = train_df[predicted_col].to_numpy()
    x_val = val_df[measured_col].to_numpy()
    y_val = val_df[predicted_col].to_numpy()

    all_vals = np.concatenate([x_train, y_train, x_val, y_val])
    lo = all_vals.min()
    hi = all_vals.max()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_train, y_train, label="train")
    plt.scatter(x_val, y_val, label="validation")
    plt.plot([lo, hi], [lo, hi], "--", label="ideal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

plot_predicted_vs_measured("memory", "pred_memory", "Measured memory (bytes)", "Predicted memory (bytes)", "Memory: predicted vs measured", "results/figures/memory_pred_vs_measured.png")
plot_predicted_vs_measured("latency", "pred_latency", "Measured latency (s)", "Predicted latency (s)", "Latency: predicted vs measured", "results/figures/latency_pred_vs_measured.png")
plot_predicted_vs_measured("energy", "pred_energy", "Measured energy (J)", "Predicted energy (J)", "Energy: predicted vs measured", "results/figures/energy_pred_vs_measured.png")

plt.figure(figsize=(8, 6))

for S_plot in [64, 128, 224, 512]:
    subset = eval_df[eval_df["S"] == S_plot].sort_values("B")
    B_plot = subset["B"].to_numpy()
    measured = subset["latency"].to_numpy()
    predicted = latency(np.full_like(B_plot, S_plot), B_plot, theta_latency)

    plt.scatter(B_plot, measured, label=f"S={S_plot} measured")
    plt.plot(B_plot, predicted, label=f"S={S_plot} predicted")

plt.xscale("log", base=2)
plt.xlabel("Batch size B")
plt.ylabel("Latency (s)")
plt.title("Latency vs batch size")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/latency_vs_batch.png", dpi=200)
plt.close()

print(json.dumps(theta, indent=2))
print("saved plots to results/figures/")