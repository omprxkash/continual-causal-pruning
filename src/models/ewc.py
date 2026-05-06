"""
Elastic Weight Consolidation (EWC).

After each task the diagonal of the empirical Fisher information matrix is
computed and stored.  During subsequent tasks an L2 regularisation term
weighted by those Fisher scores penalises drift away from the task-t optimal
parameters, thereby slowing down forgetting.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """
    Maintains a running importance estimate over all tasks seen so far.

    Usage
    -----
    ewc = EWC(model, lambda_reg=5000)

    # after training task t:
    ewc.update(model, train_loader_t, device, task_id=t)

    # during training task t+1, add to the cross-entropy loss:
    loss = ce_loss + ewc.penalty(model)
    loss.backward()
    """

    def __init__(self, model: nn.Module, lambda_reg: float = 5000.0):
        self.lambda_reg = lambda_reg
        # Cumulative Fisher diagonal and reference parameters, keyed by name
        self._fisher:  Dict[str, torch.Tensor] = {}
        self._optima:  Dict[str, torch.Tensor] = {}
        self._n_tasks_seen = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, model: nn.Module, loader: DataLoader,
               device: torch.device, task_id: int,
               n_samples: int = 1024) -> None:
        """
        Compute the diagonal Fisher for the current task and accumulate it.

        Args:
            model     : the model just trained on task `task_id`
            loader    : DataLoader for the current task's training data
            device    : torch device
            task_id   : integer task identifier (used for model.forward)
            n_samples : cap on the number of samples used for Fisher estimation
        """
        fisher_t = self._compute_fisher(model, loader, device, task_id, n_samples)

        # Accumulate (sum) Fisher matrices and store current optima
        for name, param in model.named_parameters():
            if param.requires_grad:
                f = fisher_t.get(name, torch.zeros_like(param.data))
                if name in self._fisher:
                    self._fisher[name] = self._fisher[name] + f
                else:
                    self._fisher[name] = f.clone()
                self._optima[name] = param.data.clone()

        self._n_tasks_seen += 1

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularisation term.

        Returns a scalar tensor: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
        """
        if self._n_tasks_seen == 0:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                f   = self._fisher[name].to(param.device)
                opt = self._optima[name].to(param.device)
                loss = loss + (f * (param - opt).pow(2)).sum()
        return (self.lambda_reg / 2.0) * loss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fisher(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        task_id: int,
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Diagonal empirical Fisher via gradient-squared accumulation.

        F_i = E_{x,y ~ D_t}[ (d log p(y|x,theta) / d theta_i)^2 ]
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
            x, y = x.to(device), y.to(device)
            batch = x.size(0)
            if count >= n_samples:
                break
            # Only use up to n_samples
            if count + batch > n_samples:
                x = x[: n_samples - count]
                y = y[: n_samples - count]
                batch = x.size(0)

            model.zero_grad()
            logits = model(x, task_id)
            loss   = criterion(logits, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch

            count += batch

        # Normalise
        for name in fisher:
            fisher[name] /= max(count, 1)

        model.train()
        return fisher
