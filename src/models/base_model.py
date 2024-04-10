"""
Backbone architectures for continual learning experiments.

CIFARResNet  : ResNet-18 adapted for 32×32 CIFAR input with per-task linear heads.
MNISTMlP     : Two-hidden-layer MLP for Permuted-MNIST (single shared head).
get_model    : Factory that returns the right architecture for a dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# CIFAR-adapted ResNet-18 with multi-head output
# ---------------------------------------------------------------------------

class CIFARResNet(nn.Module):
    """
    ResNet-18 for 32×32 CIFAR images in a task-incremental continual setup.

    Differences from the original ResNet-18:
      - First conv is 3×3 stride-1 (no downsampling at the stem)
      - No max-pool after the stem
      - One linear head per task (task-IL: task id is given at test time)

    Args:
        n_tasks          : total number of sequential tasks
        classes_per_task : output dimension of each task head
    """

    def __init__(self, n_tasks: int, classes_per_task: int = 5):
        super().__init__()
        self._inplanes = 64
        self.feature_dim = 512

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.heads = nn.ModuleList([
            nn.Linear(512, classes_per_task) for _ in range(n_tasks)
        ])

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self._inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self._inplanes, planes, stride, downsample)]
        self._inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[task_id](self.features(x))

    def get_backbone_params(self):
        """Parameters that belong to the shared backbone (not heads)."""
        head_ids = {id(p) for h in self.heads for p in h.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def get_head_params(self, task_id: int):
        return list(self.heads[task_id].parameters())


# ---------------------------------------------------------------------------
# Simple MLP for Permuted-MNIST
# ---------------------------------------------------------------------------

class MNISTMlP(nn.Module):
    """
    Two-hidden-layer MLP for domain-incremental Permuted-MNIST.

    Task id is not needed at inference because all tasks share the same
    10-class output head (only the input permutation changes per task).

    Args:
        input_size  : flattened input dimension (784 for MNIST)
        hidden_size : width of each hidden layer
        n_classes   : number of output classes (10 for MNIST)
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 256,
                 n_classes: int = 10):
        super().__init__()
        self.feature_dim = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_size, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(dataset: str, n_tasks: int,
              classes_per_task: int = 5) -> nn.Module:
    """
    Return the appropriate model for `dataset`.

    Args:
        dataset          : "cifar100" or "mnist"
        n_tasks          : number of sequential tasks
        classes_per_task : classes per task head (used for cifar100 only)
    """
    if dataset == "cifar100":
        return CIFARResNet(n_tasks=n_tasks, classes_per_task=classes_per_task)
    elif dataset == "mnist":
        return MNISTMlP()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'cifar100' or 'mnist'.")
