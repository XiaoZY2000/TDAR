# src/utils/seeds.py

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for:
      - Python random
      - NumPy
      - PyTorch (CPU & CUDA)

    Args:
        seed: int
        deterministic: if True, try to make CUDA behavior deterministic
                       (may reduce performance)
    """
    seed = int(seed)

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Some CUDA / CUDNN settings for reproducibility
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # allow faster but non-deterministic algorithms
        torch.backends.cudnn.benchmark = True

    # For some libraries / environments
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int):
    """
    For DataLoader with multiple workers.
    Use this in DataLoader(..., worker_init_fn=seed_worker)

    PyTorch will set:
      base_seed = torch.initial_seed()
    so we derive numpy / random seeds from it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
