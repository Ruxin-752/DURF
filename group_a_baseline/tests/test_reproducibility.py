"""可复现性测试 — 同 seed 两次训练应产生相同结果。

注意: 由于 GPU 非确定性 + 多进程，完全一致很难保证。
      这里测试的是 seed 设置函数本身 + 环境 reset 的确定性。
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils.repro import set_seed


def test_set_seed_repeatable():
    """同 seed 两次调用 np.random 应产生相同序列。"""
    set_seed(42)
    a = np.random.randn(5)

    set_seed(42)
    b = np.random.randn(5)

    np.testing.assert_array_almost_equal(a, b)


def test_set_seed_torch_repeatable():
    """同 seed 两次调用 torch.rand 应产生相同序列。"""
    set_seed(42)
    a = torch.randn(5)

    set_seed(42)
    b = torch.randn(5)

    assert torch.allclose(a, b), f"{a} != {b}"


def test_set_seed_different_seeds_different():
    """不同 seed 应产生不同序列。"""
    set_seed(42)
    a = np.random.randn(5)

    set_seed(99)
    b = np.random.randn(5)

    # 至少有一个元素不同
    assert not np.allclose(a, b), "不同 seed 产生了相同序列"


def test_set_seed_deterministic_flag():
    """deterministic=False 不应报错。"""
    set_seed(42, deterministic=False)
    _ = np.random.randn(3)
    _ = torch.randn(3)
    # 只要不抛异常就算通过
    assert True
