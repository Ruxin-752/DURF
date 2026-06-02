"""可复现性工具 — seed 设置 + 确定性配置。

用法:
    from src.utils.repro import set_seed
    set_seed(42)
"""
from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """统一设置所有随机种子，确保可复现性。

    Parameters
    ----------
    seed         : 随机种子
    deterministic: 是否启用 PyTorch 确定性算法（可能降低性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.x 额外确定性设置
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
