# Group A TAMER 交互式采集 Demo 使用方法

本文档说明如何运行 Group A baseline 的 Pygame 交互式 TAMER 数据采集界面。

## 运行环境

本项目已配置在 Conda 环境 `pantheonrl_env` 中。由于本机 `conda activate` 和 `conda run` 可能有编码/激活问题，推荐直接使用环境里的 Python：

```powershell
E:\miniconda\envs\pantheonrl_env\python.exe
```

## 启动命令

在 VS Code / Cursor 的 PowerShell 终端中运行：

```powershell
cd E:\DURF2\DURF\group_a_baseline
E:\miniconda\envs\pantheonrl_env\python.exe .\scripts\train_human_feedback_visual.py --layout cramped_room --participant P01 --total-timesteps 100000
```

## 切换地图

通过 `--layout` 参数选择地图。当前支持三个地图：

```powershell
--layout cramped_room
--layout asymmetric_advantages
--layout coordination_ring
```

示例：

```powershell
E:\miniconda\envs\pantheonrl_env\python.exe .\scripts\train_human_feedback_visual.py --layout asymmetric_advantages --participant P01 --total-timesteps 100000
```

## 操作按键

```text
WASD / 方向键 = 控制玩家厨师移动
Space         = 交互，拿/放食材或提交菜品
P             = 暂停 / 继续
Q             = TAMER 正反馈 +1
E             = TAMER 负反馈 -1
C             = coach pause，并暂停
ESC           = 退出
```

按键默认使用 `pynput` 全局监听，因此不依赖 Pygame 窗口焦点。窗口中会显示：

```text
Input: pynput
Last key
Last event
```

## 分数和反馈说明

界面中分开显示两类信号：

```text
Overcooked score
TAMER feedback sum
Feedback count
```

- `Overcooked score` 是胡闹厨房环境规则下的游戏分数，只有完成并交付一份菜时才增加。
- `Q/E` 的 `+1/-1` 是 TAMER 人类反馈信号，只用于训练 TAMER 反馈模型，不代表游戏分数。
- `training_reward` 是环境 reward 加上 TAMER feedback 后的训练信号。

## 日志输出

每次运行都会在 `logs/` 下创建一个带地图名、参与者 ID 和时间戳的目录：

```text
group_a_baseline\logs\human-feedback_<layout>_<participant>_<timestamp>\
```

主要输出文件：

```text
tamer_feedback.csv
tamer_trajectory.jsonl
```

字段说明：

- `tamer_feedback.csv`：记录 Q/E/C 的人工反馈事件。
- `tamer_trajectory.jsonl`：记录每一步 observation、action、Overcooked score、TAMER feedback 和 training reward。

## 常见问题

如果 `conda activate pantheonrl_env` 报错，可以不用修，直接使用：

```powershell
E:\miniconda\envs\pantheonrl_env\python.exe
```

如果 Pygame 窗口有焦点但按键不响应，确认窗口中显示：

```text
Input: pynput
```

如果仍然无响应，可以尝试用管理员权限打开 VS Code / Cursor 后重新运行。

如果看到 Gym 的 deprecated warning，通常可以忽略；当前环境使用旧版 Gym / SB3 兼容 PantheonRL。
