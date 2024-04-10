"""
Causal Pruning — Fisher information-guided continual learning.

Core idea
---------
After training each task, we compute an *empirical Fisher score* for every
free weight.  The Fisher score F_i = E[(∂L/∂θ_i)²] measures how much the
loss changes when θ_i is perturbed — a direct, interventional proxy for a
weight's causal importance to the current task.

Unlike PackNet, which protects weights with the largest absolute value
(a purely correlational / observational criterion), CausalPruner protects
the top-k% of free weights ranked by their Fisher score.  This captures
small-but-crucial weights that magnitude pruning would incorrectly discard
and allows large-but-redundant weights to be freed for future tasks.

Algorithm (per task t)
----------------------
1.  Train on task t with only free weights updated (via gradient masking).
2.  Compute diagonal Fisher F_i over task-t training data.
3.  For each free weight, rank by F_i (descending).
4.  Protect top keep_ratio fraction → assigned to task t, frozen hereafter.
5.  Bottom (1 - keep_ratio) fraction remain free for future tasks.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CausalPruner:
    """
    Fisher information-guided causal importance pruning.

    Args:
        keep_ratio  : fraction of *free* weights to protect per task.
                      These are the causally most important weights.
        n_samples   : number of training samples used for Fisher estimation.

    Usage
    -----
    cp = CausalPruner(keep_ratio=0.5)
    cp.register(model)

    # after training task t:
    cp.score_and_protect(model, train_loader_t, device, task_id=t)

    # during training task t+1 — mask gradients after backward:
    cp.mask_gradients(model)
    """

    def __init__(self, keep_ratio: float = 0.5, n_samples: int = 1024):
        assert 0.0 < keep_ratio < 1.0
        self.keep_ratio = keep_ratio
        self.n_samples  = n_samples

        self._free_mask:  Dict[str, torch.Tensor] = {}
        self._task_masks: Dict[int, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model: nn.Module) -> None:
        """Initialise free masks for all multi-dimensional trainable params."""
        self._free_mask = {
            name: torch.ones_like(param.data, dtype=torch.bool)
            for name, param in model.named_parameters()
            if param.requires_grad and param.dim() > 1
        }

    def score_and_protect(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        task_id: int,
    ) -> None:
        """
        Compute Fisher scores for free weights and protect the top keep_ratio
        fraction by assigning them to task_id.

        Args:
            model   : fully trained model for task `task_id`
            loader  : training DataLoader for task `task_id`
            device  : torch device
            task_id : integer task index
        """
        fisher = self._compute_fisher(model, loader, device, task_id)
        task_mask: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self._free_mask:
                    continue

                free = self._free_mask[name]
                n_free = free.sum().item()

                if n_free == 0:
                    task_mask[name] = torch.zeros_like(free)
                    continue

                f_scores = fisher.get(name, torch.zeros_like(param.data))

                # Scores of currently free weights
                free_flat_idx = free.view(-1).nonzero(as_tuple=False).squeeze(1)
                free_scores   = f_scores.view(-1)[free_flat_idx]

                n_protect = max(1, int(n_free * self.keep_ratio))
                # Highest Fisher score → most causally important → protect
                top_local_idx  = free_scores.argsort(descending=True)[:n_protect]
                protect_flat   = free_flat_idx[top_local_idx]

                # Update free mask — protected weights are no longer free
                free_updated = free.clone().view(-1)
                free_updated[protect_flat] = False
                self._free_mask[name] = free_updated.view(free.shape)

                # Record task ownership
                tm = torch.zeros(param.numel(), dtype=torch.bool)
                tm[protect_flat] = True
                task_mask[name]  = tm.view(param.shape)

        self._task_masks[task_id] = task_mask

    def mask_gradients(self, model: nn.Module) -> None:
        """
        Zero out gradients for all non-free (protected) weights.

        Call after loss.backward() and before optimizer.step() so that only
        free weights receive gradient updates.
        """
        for name, param in model.named_parameters():
            if name not in self._free_mask or param.grad is None:
                continue
            param.grad.data[~self._free_mask[name]] = 0.0

    def sparsity(self, model: nn.Module) -> float:
        """Fraction of prunable weights that are protected/frozen."""
        total, protected = 0, 0
        for name, param in model.named_parameters():
            if name not in self._free_mask:
                continue
            total     += param.numel()
            protected += (~self._free_mask[name]).sum().item()
        return protected / max(total, 1)

    def get_causal_scores(self, model: nn.Module, loader: DataLoader,
                          device: torch.device, task_id: int) -> Dict[str, torch.Tensor]:
        """Public accessor to Fisher scores for visualisation."""
        return self._compute_fisher(model, loader, device, task_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fisher(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        task_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Diagonal empirical Fisher via squared gradient accumulation.

        F_i = (1/N) Σ_{(x,y)} (∂ log p(y|x,θ) / ∂θ_i)²
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()

        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        count = 0
        for x, y in loader:
            if count >= self.n_samples:
                break
            x, y = x.to(device), y.to(device)
            batch = x.size(0)
            if count + batch > self.n_samples:
                x = x[: self.n_samples - count]
                y = y[: self.n_samples - count]
                batch = x.size(0)

            model.zero_grad()
            logits = model(x, task_id)
            loss   = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch

            count += batch

        for name in fisher:
            fisher[name] /= max(count, 1)

        model.train()
        return fisher
