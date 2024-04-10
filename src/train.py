"""
Unified continual learning training loop.

Supports five methods on two benchmark datasets via a single CLI.

Methods
-------
  finetune  : naive sequential fine-tuning (catastrophic forgetting baseline)
  ewc       : Elastic Weight Consolidation
  gem       : Approximate Gradient Episodic Memory (A-GEM)
  packnet   : magnitude-based iterative pruning
  causal    : Fisher information-guided causal pruning (proposed method)

Datasets
--------
  cifar100  : Split-CIFAR-100 (20 tasks × 5 classes)
  mnist     : Permuted-MNIST  (10 tasks)

Usage
-----
  python -m src.train --method causal --dataset cifar100 --tasks 20 \
                      --epochs 10 --lr 0.1 --batch 128 \
                      --keep_ratio 0.5 --save results/improved
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_split_cifar100, get_permuted_mnist, Task
from src.models.base_model import get_model
from src.models.ewc import EWC
from src.models.gem import GEM
from src.models.packnet import PackNet
from src.models.causal_pruning import CausalPruner
from src.evaluate import evaluate_task, build_accuracy_matrix, print_metrics


# ---------------------------------------------------------------------------
# Core training step (shared across all methods)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_id: int,
    method: str,
    ewc: "EWC | None" = None,
    gem: "GEM | None" = None,
    pnet: "PackNet | None" = None,
    cp: "CausalPruner | None" = None,
) -> float:
    model.train()
    total_loss = 0.0
    total      = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x, task_id)
        loss   = criterion(logits, y)

        if method == "ewc" and ewc is not None:
            loss = loss + ewc.penalty(model)

        loss.backward()

        # Gradient masking for pruning-based methods
        if method == "gem" and gem is not None:
            gem.project_gradient(model, device, task_id)
        if method == "packnet" and pnet is not None:
            pnet.freeze_protected(model)
        if method == "causal" and cp is not None:
            cp.mask_gradients(model)

        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

    return total_loss / max(total, 1)


# ---------------------------------------------------------------------------
# Full sequential training loop
# ---------------------------------------------------------------------------

def run_continual(
    method: str,
    dataset: str,
    n_tasks: int,
    epochs_per_task: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    # method-specific hyperparams
    ewc_lambda: float = 5000.0,
    gem_memory: int   = 200,
    prune_ratio: float = 0.5,
    keep_ratio: float  = 0.5,
    n_fisher: int      = 1024,
    seed: int          = 42,
    save_dir: str      = "results",
    verbose: bool      = True,
) -> np.ndarray:
    """
    Run sequential task training for `method` and return the accuracy matrix.

    Returns
    -------
    R : np.ndarray of shape (n_tasks, n_tasks)
        R[i, j] = accuracy on task j after training task i.
    """
    # ---- Data ---------------------------------------------------------
    if dataset == "cifar100":
        tasks = get_split_cifar100(n_tasks=n_tasks, batch_size=batch_size, seed=seed)
        classes_per_task = 100 // n_tasks
    elif dataset == "mnist":
        tasks = get_permuted_mnist(n_tasks=n_tasks, batch_size=batch_size, seed=seed)
        classes_per_task = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ---- Model --------------------------------------------------------
    model = get_model(dataset, n_tasks=n_tasks, classes_per_task=classes_per_task)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ---- Method-specific objects --------------------------------------
    ewc_obj  = EWC(model, lambda_reg=ewc_lambda) if method == "ewc"     else None
    gem_obj  = GEM(memory_per_task=gem_memory)    if method == "gem"     else None
    pnet_obj = PackNet(prune_ratio=prune_ratio)    if method == "packnet" else None
    cp_obj   = CausalPruner(keep_ratio=keep_ratio, n_samples=n_fisher) \
               if method == "causal" else None

    if pnet_obj is not None:
        pnet_obj.register(model)
    if cp_obj is not None:
        cp_obj.register(model)

    # ---- Training loop ------------------------------------------------
    per_task_results = []   # per_task_results[i] = [acc on task 0, 1, ..., i]

    for t, task in enumerate(tasks):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {t+1}/{n_tasks}  ({task.name})")
            print(f"{'='*60}")

        # Use cosine-annealing LR schedule per task
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_task)

        for epoch in range(epochs_per_task):
            t0 = time.time()
            loss = train_one_epoch(
                model, task.train_loader, optimizer, criterion, device,
                task_id=t, method=method,
                ewc=ewc_obj, gem=gem_obj, pnet=pnet_obj, cp=cp_obj,
            )
            scheduler.step()
            acc_t = evaluate_task(model, task.test_loader, device, t)
            if verbose:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1:2d}/{epochs_per_task}  "
                      f"loss={loss:.4f}  acc={acc_t*100:.2f}%  "
                      f"({elapsed:.1f}s)")

        # ---- Post-task operations ------------------------------------
        if ewc_obj is not None:
            ewc_obj.update(model, task.train_loader, device, t, n_samples=n_fisher)

        if gem_obj is not None:
            gem_obj.store_task_memory(model, task.train_loader, device, t)

        if pnet_obj is not None:
            pnet_obj.prune_and_protect(model, task_id=t)
            sparsity = pnet_obj.sparsity(model)
            if verbose:
                print(f"  [PackNet] sparsity after task {t}: {sparsity*100:.1f}%")

        if cp_obj is not None:
            cp_obj.score_and_protect(
                model, task.train_loader, device, task_id=t
            )
            sparsity = cp_obj.sparsity(model)
            if verbose:
                print(f"  [CausalPruner] sparsity after task {t}: {sparsity*100:.1f}%")

        # ---- Evaluate on all tasks seen so far -----------------------
        row = []
        for j in range(t + 1):
            acc = evaluate_task(model, tasks[j].test_loader, device, j)
            row.append(acc)
        per_task_results.append(row)

        if verbose:
            print(f"\n  Accuracies on tasks 0..{t}: "
                  + "  ".join(f"{a*100:.1f}%" for a in row))

    # ---- Build and save accuracy matrix ------------------------------
    R = build_accuracy_matrix(per_task_results)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{method}_{dataset}_acc_matrix.npy")
    np.save(out_path, R)
    if verbose:
        print(f"\nSaved accuracy matrix → {out_path}")
        print_metrics(R, method=f"{method.upper()} on {dataset}")

    return R


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Continual learning benchmark runner."
    )
    p.add_argument("--method", choices=["finetune", "ewc", "gem", "packnet", "causal"],
                   default="causal", help="CL strategy to run")
    p.add_argument("--dataset", choices=["cifar100", "mnist"],
                   default="cifar100")
    p.add_argument("--tasks",   type=int, default=20,
                   help="Number of sequential tasks")
    p.add_argument("--epochs",  type=int, default=10,
                   help="Epochs per task")
    p.add_argument("--lr",      type=float, default=0.1)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--save",    type=str, default="results",
                   help="Directory to save accuracy matrix .npy files")
    # EWC
    p.add_argument("--ewc_lambda", type=float, default=5000.0)
    # GEM
    p.add_argument("--gem_memory", type=int, default=200)
    # PackNet
    p.add_argument("--prune_ratio", type=float, default=0.5)
    # Causal
    p.add_argument("--keep_ratio", type=float, default=0.5)
    p.add_argument("--n_fisher",   type=int,   default=1024)
    return p.parse_args()


def main():
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: {args.method}  |  Dataset: {args.dataset}  "
          f"|  Tasks: {args.tasks}  |  Epochs/task: {args.epochs}")

    run_continual(
        method=args.method,
        dataset=args.dataset,
        n_tasks=args.tasks,
        epochs_per_task=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        device=device,
        ewc_lambda=args.ewc_lambda,
        gem_memory=args.gem_memory,
        prune_ratio=args.prune_ratio,
        keep_ratio=args.keep_ratio,
        n_fisher=args.n_fisher,
        seed=args.seed,
        save_dir=args.save,
    )


if __name__ == "__main__":
    main()
