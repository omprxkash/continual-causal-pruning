"""
Generate all comparison figures from saved accuracy matrices.

Expected inputs (produced by src/train.py):
  results/baseline/finetune_cifar100_acc_matrix.npy
  results/baseline/ewc_cifar100_acc_matrix.npy
  results/baseline/gem_cifar100_acc_matrix.npy
  results/improved/packnet_cifar100_acc_matrix.npy
  results/improved/causal_cifar100_acc_matrix.npy

Outputs:
  results/baseline/accuracy_matrix_<method>.png   — per-method heatmap
  results/forgetting_curves.png                    — accuracy on task-0 over time
  results/bwt_fwt_comparison.png                  — grouped bar chart
  results/sparsity_vs_accuracy.png                 — scatter
  results/benchmark_comparison.png                 — 4-panel figure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import backward_transfer, forward_transfer, average_accuracy

RESULTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helper to load one matrix
# ---------------------------------------------------------------------------

def load_matrix(path: str) -> np.ndarray | None:
    p = Path(path)
    if p.exists():
        return np.load(p)
    print(f"  [warn] {p} not found — skipping")
    return None


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "Fine-tuning":     RESULTS_DIR / "baseline" / "finetune_cifar100_acc_matrix.npy",
    "EWC":             RESULTS_DIR / "baseline" / "ewc_cifar100_acc_matrix.npy",
    "GEM":             RESULTS_DIR / "baseline" / "gem_cifar100_acc_matrix.npy",
    "PackNet":         RESULTS_DIR / "improved" / "packnet_cifar100_acc_matrix.npy",
    "CausalPruning":   RESULTS_DIR / "improved" / "causal_cifar100_acc_matrix.npy",
}

COLORS = {
    "Fine-tuning":   "#e74c3c",
    "EWC":           "#f39c12",
    "GEM":           "#3498db",
    "PackNet":       "#2ecc71",
    "CausalPruning": "#9b59b6",
}

SPARSITY = {
    "Fine-tuning":   0.0,
    "EWC":           0.0,
    "GEM":           0.0,
    "PackNet":       0.50,
    "CausalPruning": 0.50,
}


# ---------------------------------------------------------------------------
# 1. Per-method accuracy matrix heatmap
# ---------------------------------------------------------------------------

def plot_accuracy_matrix(R: np.ndarray, method: str, out_path: str) -> None:
    T = R.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, T * 0.6), max(6, T * 0.5)))

    mask = np.isnan(R)
    r_display = np.where(mask, 0.0, R * 100)

    im = ax.imshow(r_display, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{j}" for j in range(T)], fontsize=8)
    ax.set_yticklabels([f"T{i}" for i in range(T)], fontsize=8)
    ax.set_xlabel("Task evaluated", fontsize=11)
    ax.set_ylabel("After training task", fontsize=11)
    ax.set_title(f"{method} — Accuracy Matrix", fontsize=13, fontweight="bold")

    for i in range(T):
        for j in range(T):
            if not mask[i, j]:
                ax.text(j, i, f"{R[i,j]*100:.0f}",
                        ha="center", va="center",
                        fontsize=6, color="black" if R[i,j] > 0.5 else "white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# 2. Forgetting curves (accuracy on task 0 after each successive task)
# ---------------------------------------------------------------------------

def plot_forgetting_curves(matrices: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, R in matrices.items():
        if R is None:
            continue
        T = R.shape[0]
        # Accuracy on task 0 after training task i (i=0,1,...,T-1)
        curve = [R[i, 0] * 100 for i in range(T)]
        ax.plot(range(T), curve, marker="o", label=name,
                color=COLORS.get(name, "gray"), linewidth=2, markersize=5)

    ax.set_xlabel("Number of tasks seen", fontsize=12)
    ax.set_ylabel("Accuracy on Task 0 (%)", fontsize=12)
    ax.set_title("Catastrophic Forgetting Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# 3. BWT / FWT comparison bar chart
# ---------------------------------------------------------------------------

def plot_bwt_fwt(matrices: dict, out_path: str) -> None:
    names, bwts, fwts, avgs = [], [], [], []

    for name, R in matrices.items():
        if R is None:
            continue
        names.append(name)
        bwts.append(backward_transfer(R) * 100)
        fwts.append(forward_transfer(R) * 100)
        avgs.append(average_accuracy(R) * 100)

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width,  avgs, width, label="Avg Accuracy (%)", color="#2ecc71", alpha=0.85)
    b2 = ax.bar(x,          bwts, width, label="BWT (%)",           color="#e74c3c", alpha=0.85)
    b3 = ax.bar(x + width,  fwts, width, label="FWT (%)",           color="#3498db", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Average Accuracy / BWT / FWT Comparison", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# 4. Sparsity vs Accuracy scatter
# ---------------------------------------------------------------------------

def plot_sparsity_accuracy(matrices: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, R in matrices.items():
        if R is None:
            continue
        acc  = average_accuracy(R) * 100
        sp   = SPARSITY.get(name, 0.0) * 100
        ax.scatter(sp, acc, s=120, color=COLORS.get(name, "gray"),
                   label=name, zorder=5)
        ax.annotate(name, (sp, acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Network Sparsity (%)", fontsize=12)
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.set_title("Sparsity vs Average Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# 5. 4-panel benchmark comparison
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(matrices: dict, out_path: str) -> None:
    names = [n for n, R in matrices.items() if R is not None]
    Rs    = [matrices[n] for n in names]

    avgs = [average_accuracy(R) * 100 for R in Rs]
    bwts = [backward_transfer(R) * 100 for R in Rs]
    fwts = [forward_transfer(R) * 100 for R in Rs]
    sps  = [SPARSITY.get(n, 0.0) * 100 for n in names]

    clrs = [COLORS.get(n, "gray") for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Continual Learning Benchmark Comparison\n(Split-CIFAR-100, 20 tasks)",
                 fontsize=14, fontweight="bold")

    def bar_panel(ax, values, title, ylabel, color_list):
        bars = ax.bar(x, values, color=color_list, alpha=0.85, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    bar_panel(axes[0, 0], avgs, "Average Accuracy (%)", "Acc (%)", clrs)
    bar_panel(axes[0, 1], bwts, "Backward Transfer (%) — lower is better forgetting", "BWT (%)", clrs)
    bar_panel(axes[1, 0], fwts, "Forward Transfer (%)", "FWT (%)", clrs)

    # Panel 4: sparsity vs accuracy scatter
    for name, acc, sp in zip(names, avgs, sps):
        axes[1, 1].scatter(sp, acc, s=150, color=COLORS.get(name, "gray"),
                           label=name, zorder=5)
        axes[1, 1].annotate(name, (sp, acc), textcoords="offset points",
                            xytext=(5, 4), fontsize=8)
    axes[1, 1].set_xlabel("Sparsity (%)", fontsize=11)
    axes[1, 1].set_ylabel("Avg Accuracy (%)", fontsize=11)
    axes[1, 1].set_title("Sparsity vs Accuracy", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading accuracy matrices...")
    matrices = {}
    for name, path in METHODS.items():
        R = load_matrix(str(path))
        matrices[name] = R

    available = {k: v for k, v in matrices.items() if v is not None}

    if not available:
        print("No result files found. Run src/train.py for each method first.")
        return

    print(f"\nGenerating plots for: {list(available.keys())}\n")

    # Per-method heatmaps
    for name, R in available.items():
        subdir = "baseline" if name in ("Fine-tuning", "EWC", "GEM") else "improved"
        out = RESULTS_DIR / subdir / f"accuracy_matrix_{name.lower().replace(' ', '_')}.png"
        plot_accuracy_matrix(R, name, str(out))

    # Forgetting curves
    plot_forgetting_curves(
        matrices,
        str(RESULTS_DIR / "forgetting_curves.png"),
    )

    # BWT / FWT bar chart
    plot_bwt_fwt(
        matrices,
        str(RESULTS_DIR / "bwt_fwt_comparison.png"),
    )

    # Sparsity vs accuracy
    plot_sparsity_accuracy(
        matrices,
        str(RESULTS_DIR / "sparsity_vs_accuracy.png"),
    )

    # 4-panel benchmark comparison
    plot_benchmark_comparison(
        matrices,
        str(RESULTS_DIR / "benchmark_comparison.png"),
    )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
