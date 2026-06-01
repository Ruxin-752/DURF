# PantheonRL + Overcooked-AI 环境搭建指南

## ⚠️ 重要: 环境隔离说明

**PantheonRL 需要独立的环境，不能与 `requirements.txt` 共用！**

| 用途 | 环境 | 核心依赖 |
|------|------|----------|
| **PantheonRL 训练** (本指南) | 独立环境 (如 `pantheonrl_env`) | `gym==0.25.2`, `stable-baselines3`, `shimmy` |
| **LLM 实验** (`requirements.txt`) | 另一环境 (如 `llm_env`) | `gymnasium==1.2.3`, `openai` |

原因: PantheonRL 基于旧版 OpenAI Gym，而 `requirements.txt` 用 Gymnasium，两者 API 不兼容。

---

## 文件结构

```
overcooked_ai/
├── patches/
│   ├── pantheonrl_overcooked.patch   # PantheonRL 代码修复
│   └── gym_numpy_compat.patch        # Gym NumPy 2.0 兼容修复
├── setup_pantheonrl.bat              # 一键搭建脚本
└── PANTHEONRL_SETUP.md               # 本说明文件
```

## 一键搭建 (推荐)

在 `overcooked_ai` 根目录下双击或运行:

```bat
setup_pantheonrl.bat
```

脚本会自动检测 GPU、创建环境、安装依赖、应用 patch。

## 手动搭建步骤

```bash
# 0. 创建并激活独立环境
conda create -n pantheonrl_env python=3.10 -y
conda activate pantheonrl_env

# 1. 安装 PyTorch
#    有 NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    无 GPU / CPU only:
pip install torch torchvision torchaudio

# 2. 安装 overcooked_ai (在 overcooked_ai 目录下)
pip install -e .

# 3. 安装 PantheonRL (clone 到 overcooked_ai 同级目录)
git clone https://github.com/Stanford-ILIAD/PantheonRL.git ../PantheonRL
pip install -e ../PantheonRL

# 4. 安装 gym + NumPy + shimmy
#    固定 numpy<2.0: 避免 np.bool8 被 NumPy 2.0 移除的问题, 无需 patch gym
pip install "gym<0.26" "numpy<2.0" "shimmy>=2.0"

# 5. 应用 PantheonRL patch (-p2 去掉 patch 内多余的路径前缀)
cd ../PantheonRL
git apply -p2 ../overcooked_ai/patches/pantheonrl_overcooked.patch
```

## 运行训练

```bash
conda activate pantheonrl_env
cd PantheonRL

python trainer.py OvercookedMultiEnv-v0 PPO PPO ^
    --env-config "{\"layout_name\":\"cramped_room\"}" ^
    --ego-config "{\"verbose\":1}" ^
    --total-timesteps 10000 --seed 42
```

## 修复内容说明

### Patch 1: `pantheonrl_overcooked.patch`

| 文件 | 修改 | 原因 |
|------|------|------|
| `overcookedgym/overcooked.py` | 添加 `seed()` 方法 | Gym 0.25+ 的 `reset()` 需要 `seed()` 方法 |
| `overcookedgym/overcooked.py` | `shaped_r` → `shaped_r_by_agent[0]` | 新版 overcooked_ai API 变更 |
| `pantheonrl/common/agents.py` | `dones` 参数: `bool` → `np.array` | SB3 的 `compute_returns_and_advantage` 期望数组 |

### Patch 2: `gym_numpy_compat.patch`

| 文件 | 修改 | 原因 |
|------|------|------|
| `gym/utils/passive_env_checker.py` | `np.bool8` → `np.bool_` | NumPy 2.0 移除了 `np.bool8` |

## 环境要求

- Python 3.10
- PyTorch 2.x (CUDA 12.1 / CPU 均可)
- Gym 0.25.2 (非 Gymnasium!)
- NumPy < 2.0 (锁版本避免 np.bool8 兼容问题)
