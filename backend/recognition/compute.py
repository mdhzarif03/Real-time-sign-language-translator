from __future__ import annotations

import os

import torch


def configure_torch_threads() -> int:
    """Use a bounded CPU thread pool consistently in training and inference."""
    requested = int(os.getenv("SIGNFLOW_TORCH_THREADS", "4"))
    if not 1 <= requested <= 64:
        raise ValueError("SIGNFLOW_TORCH_THREADS must be between 1 and 64")
    torch.set_num_threads(requested)
    return requested
