# =============================================================================
# src/utils.py
#
# Utility helpers: reproducibility seeds, directory setup, device selection.
# =============================================================================

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes cuDNN deterministic (small speed cost — acceptable for demos)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Utils] Using device: {device}")
    return device


def make_output_dirs(*dirs: str) -> None:
    """Create all output directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
