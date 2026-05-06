"""
Dataset preparation for continual learning benchmarks.

Split-CIFAR-100  : CIFAR-100 partitioned into n_tasks disjoint class groups.
Permuted-MNIST   : MNIST with a different fixed pixel permutation per task.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Task container
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: int
    train_loader: DataLoader
    test_loader: DataLoader
    classes: List[int]          # original class labels in this task
    n_classes: int              # number of classes (= len(classes))
    name: str = ""


# ---------------------------------------------------------------------------
# Internal dataset wrappers
# ---------------------------------------------------------------------------

class _RemappedSubset(Dataset):
    """Subset of a dataset restricted to `indices`, labels remapped to 0..C-1."""

    def __init__(self, dataset: Dataset, indices: List[int], class_map: dict):
        self.dataset = dataset
        self.indices = indices
        self.class_map = class_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.class_map[int(label)]


class _PermutedDataset(Dataset):
    """Applies a fixed pixel permutation to every image in `dataset`."""

    def __init__(self, dataset: Dataset, permutation: np.ndarray):
        self.dataset = dataset
        self.perm = torch.from_numpy(permutation).long()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        flat = img.reshape(-1)
        flat = flat[self.perm]
        return flat.reshape(img.shape), label


# ---------------------------------------------------------------------------
# Split-CIFAR-100
# ---------------------------------------------------------------------------

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_split_cifar100(
    n_tasks: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    data_root: str = "./data",
    num_workers: int = 2,
) -> List[Task]:
    """
    Returns a list of Task objects for Split-CIFAR-100.

    CIFAR-100 has 100 classes.  We partition them into n_tasks non-overlapping
    groups of (100 // n_tasks) classes each, shuffled by `seed`.

    Within each task labels are remapped to 0 .. classes_per_task-1 so that
    task-specific linear heads each have `classes_per_task` output neurons.
    """
    assert 100 % n_tasks == 0, "n_tasks must divide 100"
    classes_per_task = 100 // n_tasks

    rng = np.random.default_rng(seed)
    class_order = rng.permutation(100).tolist()

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    train_full = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=tf_train
    )
    test_full = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=tf_test
    )

    # Pre-collect per-class indices once to avoid O(N*C) repeated scans
    train_labels = np.array(train_full.targets)
    test_labels  = np.array(test_full.targets)

    tasks: List[Task] = []
    for t in range(n_tasks):
        task_classes = class_order[t * classes_per_task : (t + 1) * classes_per_task]
        class_map    = {cls: i for i, cls in enumerate(task_classes)}

        tr_idx = np.where(np.isin(train_labels, task_classes))[0].tolist()
        te_idx = np.where(np.isin(test_labels,  task_classes))[0].tolist()

        tr_loader = DataLoader(
            _RemappedSubset(train_full, tr_idx, class_map),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        te_loader = DataLoader(
            _RemappedSubset(test_full, te_idx, class_map),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        tasks.append(Task(
            task_id=t,
            train_loader=tr_loader,
            test_loader=te_loader,
            classes=task_classes,
            n_classes=classes_per_task,
            name=f"CIFAR100-T{t}",
        ))

    return tasks


# ---------------------------------------------------------------------------
# Permuted-MNIST
# ---------------------------------------------------------------------------

_MNIST_MEAN = (0.1307,)
_MNIST_STD  = (0.3081,)


def get_permuted_mnist(
    n_tasks: int = 10,
    batch_size: int = 256,
    seed: int = 42,
    data_root: str = "./data",
    num_workers: int = 2,
) -> List[Task]:
    """
    Returns a list of Task objects for Permuted-MNIST.

    Task 0 uses the identity permutation (original MNIST).
    Tasks 1..n_tasks-1 each use a distinct random pixel permutation.
    All tasks share the same 10 output classes (domain-incremental setting).
    """
    rng = np.random.default_rng(seed)

    tf_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])

    train_full = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=tf_base
    )
    test_full = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=tf_base
    )

    tasks: List[Task] = []
    for t in range(n_tasks):
        perm = np.arange(784) if t == 0 else rng.permutation(784)

        tr_loader = DataLoader(
            _PermutedDataset(train_full, perm),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        te_loader = DataLoader(
            _PermutedDataset(test_full, perm),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        tasks.append(Task(
            task_id=t,
            train_loader=tr_loader,
            test_loader=te_loader,
            classes=list(range(10)),
            n_classes=10,
            name=f"PMNIST-T{t}",
        ))

    return tasks
