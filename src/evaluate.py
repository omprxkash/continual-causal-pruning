"""
Evaluation utilities for continual learning.

Key metrics
-----------
Accuracy matrix R[i][j] : accuracy on task j immediately after training task i.
                          R[i][i] is the just-trained task's accuracy.
                          R[i][j] for j < i measures retention (backward transfer).
                          R[i][j] for j > i can measure forward transfer potential.

BWT (Backward Transfer) : average drop in accuracy on old tasks after all tasks trained.
    BWT = (1/(T-1)) * sum_{i=0}^{T-2} ( R[T-1][i] - R[i][i] )
    Negative BWT means forgetting.

FWT (Forward Transfer)  : average accuracy on a task before it is trained,
    relative to a random-init baseline.
    FWT = (1/(T-1)) * sum_{i=1}^{T-1} ( R[i-1][i] - b[i] )
    where b[i] is zero-shot (random init) accuracy on task i.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Per-task accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_id: int,
) -> float:
    """Return top-1 accuracy (fraction) of `model` on the given DataLoader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, task_id)
        preds  = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Full accuracy matrix
# ---------------------------------------------------------------------------

def build_accuracy_matrix(
    per_task_results: List[List[float]],
) -> np.ndarray:
    """
    Convert a list-of-lists of per-task accuracies into a numpy matrix.

    `per_task_results[i]` must contain the accuracy on every task 0..T-1
    immediately after training task i.  Entries for tasks not yet trained
    (future tasks) are set to NaN.

    Returns
    -------
    R : np.ndarray of shape (T, T)
        R[i, j] = accuracy on task j after training task i.
        R[i, j] = NaN if j > i  (task j not yet seen when evaluating after task i).
    """
    T = len(per_task_results)
    R = np.full((T, T), np.nan)
    for i, row in enumerate(per_task_results):
        for j, acc in enumerate(row):
            if j <= i:
                R[i, j] = acc
    return R


# ---------------------------------------------------------------------------
# BWT and FWT
# ---------------------------------------------------------------------------

def backward_transfer(R: np.ndarray) -> float:
    """
    Backward Transfer (BWT) — average forgetting.

    BWT = (1/(T-1)) * sum_{i=0}^{T-2} [ R[T-1, i] - R[i, i] ]

    Negative BWT indicates forgetting; positive is forward plasticity.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    diffs = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    return float(np.mean(diffs))


def forward_transfer(R: np.ndarray, random_acc: Optional[List[float]] = None) -> float:
    """
    Forward Transfer (FWT) — how much past training helps future tasks.

    FWT = (1/(T-1)) * sum_{i=1}^{T-1} [ R[i-1, i] - b[i] ]

    where b[i] is the random-init (zero-shot) accuracy on task i.
    If `random_acc` is None, b[i] = 1/n_classes (uniform guess).
    """
    T = R.shape[0]
    if T < 2:
        return 0.0

    fwts = []
    for i in range(1, T):
        # R[i-1, i] is the accuracy on task i BEFORE training it
        # (if it was evaluated — for strict task-IL it is typically evaluated)
        r_before = R[i - 1, i] if not np.isnan(R[i - 1, i]) else 0.0
        b = random_acc[i] if random_acc is not None else 0.0
        fwts.append(r_before - b)

    return float(np.mean(fwts))


def average_accuracy(R: np.ndarray) -> float:
    """Mean accuracy on all tasks evaluated after training the final task."""
    T = R.shape[0]
    return float(np.nanmean(R[T - 1, :T]))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_metrics(R: np.ndarray, method: str = "") -> None:
    T = R.shape[0]
    aa  = average_accuracy(R)
    bwt = backward_transfer(R)
    fwt = forward_transfer(R)

    header = f"=== {method} ===" if method else "=== Results ==="
    print(header)
    print(f"  Average Accuracy (after all tasks) : {aa*100:.2f}%")
    print(f"  Backward Transfer (BWT)            : {bwt*100:.2f}%")
    print(f"  Forward Transfer  (FWT)            : {fwt*100:.2f}%")
    print()
    print("  Accuracy Matrix (rows=after task i, cols=task j):")
    header_row = "        " + "  ".join(f"T{j:02d}" for j in range(T))
    print(header_row)
    for i in range(T):
        row = f"  T{i:02d} : "
        row += "  ".join(
            f"{R[i,j]*100:5.1f}" if not np.isnan(R[i, j]) else "  nan"
            for j in range(T)
        )
        print(row)
    print()
