import os
import random

import numpy as np
import torch

SEED = 6 * 9
ULTIMATE_SEED = int(np.base_repr(SEED, base=13))

def set_seed(seed=ULTIMATE_SEED):
    """Seed python, numpy, and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favor reproducibility over the last few % of cuDNN autotuner speed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_device():
    """Return the CUDA device when available (RTX 5090 here), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
