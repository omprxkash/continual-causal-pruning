"""
PackNet — structured pruning for task-incremental continual learning.

After training each task, a fraction of the *currently free* weights (those
not yet claimed by any previous task) are pruned away permanently.  The
surviving free weights are then frozen and assigned to the current task.
Subsequent tasks can only modify weights that are still free.

This creates non-overlapping binary sub-networks, one per task, with zero
interference between tasks.  The cost is reduced network capacity over time.

Reference: Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single
Network by Iterative Pruning", CVPR 2018.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PackNet:
    """
    Magnitude-based iterative pruning for continual learning.

    Args:
        prune_ratio : fraction of *free* weights to prune after each task
                      (remaining free weights become the task's sub-network)

    Usage
    -----
    pn = PackNet(prune_ratio=0.5)

    # At the very start, register the model once:
    pn.register(model)

    # After training task t (before moving to t+1):
    pn.prune_and_protect(model, task_id=t)

    # During training task t+1: call at each backward step to zero out
    # gradients for protected weights:
    pn.freeze_protected(model)
    """

    def __init__(self, prune_ratio: float = 0.5):
        assert 0.0 < prune_ratio < 1.0
        self.prune_ratio = prune_ratio

        # Boolean masks (True = weight is free / available for training)
        self._free_mask:  Dict[str, torch.Tensor] = {}
        # Per-task binary masks (True = this weight belongs to task t)
        self._task_masks: Dict[int, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model: nn.Module) -> None:
        """Initialise free masks for all trainable parameters."""
        self._free_mask = {
            name: torch.ones_like(param.data, dtype=torch.bool)
            for name, param in model.named_parameters()
            if param.requires_grad and param.dim() > 1  # skip 1-D BN params
        }

    def prune_and_protect(self, model: nn.Module, task_id: int) -> None:
        """
        1. Among the currently free weights, prune the `prune_ratio` fraction
           with the smallest absolute value (set them permanently to zero and
           mark them as no longer free).
        2. The remaining free weights survive and are assigned to `task_id`
           (they will be frozen from now on).
        """
        task_mask: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self._free_mask:
                    continue

                free = self._free_mask[name]          # bool mask
                free_vals = param.data[free].abs()    # magnitudes of free weights

                n_free = free_vals.numel()
                if n_free == 0:
                    task_mask[name] = torch.zeros_like(free)
                    continue

                # How many weights to prune
                n_prune = max(1, int(n_free * self.prune_ratio))
                # Indices of free weights sorted by magnitude (ascending)
                sorted_idx = free_vals.argsort()

                # Build flat indices into the full parameter tensor
                free_flat_idx = free.view(-1).nonzero(as_tuple=False).squeeze(1)
                prune_flat_idx   = free_flat_idx[sorted_idx[:n_prune]]
                protect_flat_idx = free_flat_idx[sorted_idx[n_prune:]]

                # Zero out pruned weights permanently
                param.data.view(-1)[prune_flat_idx] = 0.0

                # Mark pruned and protected weights as no longer free
                free_updated = free.clone().view(-1)
                free_updated[prune_flat_idx]   = False
                free_updated[protect_flat_idx] = False
                self._free_mask[name] = free_updated.view(free.shape)

                # Record which weights belong to this task
                tm = torch.zeros(param.numel(), dtype=torch.bool)
                tm[protect_flat_idx] = True
                task_mask[name] = tm.view(param.shape)

        self._task_masks[task_id] = task_mask

    def freeze_protected(self, model: nn.Module) -> None:
        """
        Zero out gradients for all non-free parameters so that only
        currently free weights receive gradient updates.

        Call this *after* loss.backward() and *before* optimizer.step().
        """
        for name, param in model.named_parameters():
            if name not in self._free_mask or param.grad is None:
                continue
            # Only free weights should get non-zero gradients
            param.grad.data[~self._free_mask[name]] = 0.0

    # ------------------------------------------------------------------
    # Utility / diagnostics
    # ------------------------------------------------------------------

    def sparsity(self, model: nn.Module) -> float:
        """Fraction of trainable weights that are frozen (not free)."""
        total, frozen = 0, 0
        for name, param in model.named_parameters():
            if name not in self._free_mask:
                continue
            total  += param.numel()
            frozen += (~self._free_mask[name]).sum().item()
        return frozen / max(total, 1)

    def n_free_weights(self, model: nn.Module) -> int:
        total = 0
        for name, param in model.named_parameters():
            if name in self._free_mask:
                total += self._free_mask[name].sum().item()
        return int(total)
