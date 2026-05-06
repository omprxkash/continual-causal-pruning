"""
Gradient Episodic Memory (A-GEM variant).

The original GEM (Lopez-Paz & Ranzato, NeurIPS 2017) stores exemplars from
every past task and solves a constrained QP to find a gradient that does not
increase the loss on any past task.  Here we implement A-GEM (Chaudhry et al.,
ICLR 2019), which is computationally identical but uses the *average* gradient
over all past exemplars as a single reference vector, making the projection a
simple closed-form operation with no QP solver required.

A-GEM projection rule:
    g_ref = mean gradient over all episodic memories
    if g · g_ref < 0:
        g = g - (g · g_ref / ||g_ref||²) * g_ref
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GEM:
    """
    A-GEM continual learning strategy.

    Usage
    -----
    gem = GEM(memory_per_task=200)

    # after training task t, store episodic memory:
    gem.store_task_memory(model, train_loader_t, device, task_id=t)

    # during training task t+1, after loss.backward():
    gem.project_gradient(model)
    optimizer.step()
    """

    def __init__(self, memory_per_task: int = 200):
        self.memory_per_task = memory_per_task
        # Stored (X, Y) tensors per task (on CPU to save GPU memory)
        self._memory: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_task_memory(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        task_id: int,
    ) -> None:
        """Randomly sample `memory_per_task` examples from `loader` and store them."""
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)

        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)

        n = X.size(0)
        if n > self.memory_per_task:
            idx = torch.randperm(n)[: self.memory_per_task]
            X, Y = X[idx], Y[idx]

        self._memory[task_id] = (X.cpu(), Y.cpu())

    def project_gradient(self, model: nn.Module,
                         device: torch.device,
                         task_id: int) -> None:
        """
        After loss.backward() on the current task, check whether the current
        gradient conflicts with the episodic reference gradient.  If so,
        project the current gradient.
        """
        if not self._memory:
            return  # no past tasks yet

        # Compute reference gradient on a random batch from all past memories
        ref_grad = self._compute_reference_gradient(model, device, task_id)
        if ref_grad is None:
            return

        # Collect current gradient as a flat vector
        cur_grad = self._get_flat_grad(model)

        # Check constraint: g · g_ref ≥ 0
        dot = torch.dot(cur_grad, ref_grad)
        if dot < 0:
            # Project: g = g - (g·g_ref / ||g_ref||²) * g_ref
            ref_norm_sq = ref_grad.dot(ref_grad) + 1e-12
            cur_grad = cur_grad - (dot / ref_norm_sq) * ref_grad
            self._set_flat_grad(model, cur_grad)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reference_gradient(
        self,
        model: nn.Module,
        device: torch.device,
        current_task_id: int,
    ) -> Optional[torch.Tensor]:
        past_ids = [tid for tid in self._memory if tid != current_task_id]
        if not past_ids:
            return None

        criterion = nn.CrossEntropyLoss()
        saved_grads = []

        for tid in past_ids:
            X_mem, Y_mem = self._memory[tid]
            # Sample a mini-batch from memory
            n = X_mem.size(0)
            idx = torch.randperm(n)[:min(n, 64)]
            x_b = X_mem[idx].to(device)
            y_b = Y_mem[idx].to(device)

            model.zero_grad()
            logits = model(x_b, tid)
            loss   = criterion(logits, y_b)
            loss.backward()

            saved_grads.append(self._get_flat_grad(model).clone())

        model.zero_grad()
        # A-GEM: use the *average* reference gradient
        return torch.stack(saved_grads).mean(dim=0)

    @staticmethod
    def _get_flat_grad(model: nn.Module) -> torch.Tensor:
        grads = []
        for p in model.parameters():
            if p.requires_grad:
                g = p.grad if p.grad is not None else torch.zeros_like(p.data)
                grads.append(g.view(-1))
        return torch.cat(grads)

    @staticmethod
    def _set_flat_grad(model: nn.Module, flat_grad: torch.Tensor) -> None:
        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.grad = flat_grad[offset: offset + numel].view_as(p.data).clone()
                offset += numel
