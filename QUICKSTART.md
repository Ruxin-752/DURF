# 环境配置快速上手（从零开始）

> **适用对象**：Windows 电脑，尚未安装任何开发工具

---

## 第一步：安装前置工具（只需做一次）

### 1. 安装 Git

下载并安装 Git for Windows：https://git-scm.com/download/win

安装时全部默认选项即可。

### 2. 安装 Miniconda

下载并安装 Miniconda（Python 环境管理器）：https://docs.anaconda.com/miniconda/install/#windows

安装时勾选 **"Add Miniconda3 to my PATH environment variable"**（推荐勾选，方便在普通终端使用）。

安装完成后，打开 **Anaconda Prompt**（开始菜单搜索即可找到）。

---

## 第二步：Clone 仓库

在 Anaconda Prompt 中运行：

```bat
git clone https://github.com/<你的GitHub用户名>/overcooked_ai.git
cd overcooked_ai
```

> 把 `<你的GitHub用户名>` 替换为实际地址（问 Ruxin 要链接）。

---

## 第三步：一键配置环境

在 Anaconda Prompt 中，确保当前在 `overcooked_ai` 目录下，运行：

```bat
setup_pantheonrl.bat
```

脚本会自动完成所有步骤（约 5-10 分钟，取决于网速）：
- 创建独立 conda 环境 `pantheonrl_env`
- 安装 PyTorch（自动检测是否有 GPU）
- 安装 overcooked_ai、PantheonRL、gym 等依赖
- 应用兼容性 patch

---

## 第四步：运行训练测试

```bat
conda activate pantheonrl_env
cd ..\PantheonRL
python trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config "{\"layout_name\":\"cramped_room\"}" --ego-config "{\"verbose\":1}" --total-timesteps 1000 --seed 42
```

看到训练 log 输出即表示环境配置成功。

---

## 遇到问题？

| 错误信息 | 解决方法 |
|----------|----------|
| `conda: command not found` | 用 **Anaconda Prompt** 运行脚本，不要用普通 cmd |
| `git: command not found` | Git 没装好，重装时确认勾选了 "Add to PATH" |
| `激活环境失败` | 在 Anaconda Prompt 运行 `conda init cmd.exe`，关闭重开再试 |
| `Clone PantheonRL 失败` | 检查网络，或挂代理后重试 |
| `验证失败` | 把终端输出截图发给 Ruxin |

---

## 文件说明

```
overcooked_ai/
├── QUICKSTART.md              ← 本文件
├── PANTHEONRL_SETUP.md        ← 详细手动步骤（可选阅读）
├── setup_pantheonrl.bat       ← 一键配置脚本
└── patches/
    ├── pantheonrl_overcooked.patch   ← PantheonRL 兼容性修复
    └── gym_numpy_compat.patch        ← 备用（当前通过 numpy 版本锁处理）
```
