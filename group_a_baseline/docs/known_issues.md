# 已知问题与 Workaround

---

## 1. gym 0.25 兼容性警告

**问题**: PantheonRL 的 `OvercookedMultiEnv` 基于 gym 0.25（旧 API），SB3 2.x 和 NumPy 2.0 会打印大量 deprecation warnings。

```
Gym has been unmaintained since 2022 and does not support NumPy 2.0...
WARN: Initializing wrapper in old step API which returns one bool instead of two...
```

**影响**: 不影响功能，但输出杂乱。

**Workaround**:
- 设置环境变量 `PYTHONWARNINGS=ignore` 屏蔽警告
- 或升级 PantheonRL 到 gymnasium（长期方案）

---

## 2. PPO GPU 警告 (MlpPolicy)

**问题**: 使用 `MlpPolicy` 时 SB3 建议使用 CPU。

```
You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU...
```

**影响**: GPU 也能跑，但利用率低。对于 Overcooked 的小规模网络，CPU 可能更快。

**Workaround**: 在 config 中设置 `device: cpu`，或启动时加 `--device cpu`。

---

## 3. Windows 路径分隔符

**问题**: 代码中硬编码了 `/` 作为路径分隔符，Windows 下需要 `\`。

**影响**: `config/default.yaml` 等相对路径在 Windows 上可能找不到。

**Workaround**: 已使用 `Path` 对象处理路径，但启动脚本需要 `cd` 到项目根目录。

---

## 4. keyboard_listener 需要控制台焦点

**问题**: `keyboard_listener.py` 使用 `pynput` 监听全局键盘事件，需要终端窗口有焦点。

**影响**: 如果训练在后台运行，键盘输入不会被捕获。

**Workaround**: 确保训练窗口在前台。未来可考虑使用游戏窗口的按键事件。

---

## 5. 多进程在 Windows 上的限制

**问题**: Windows 缺少 `fork()`，`multiprocessing.Process` 需要 `freeze_support()` 保护。

**影响**: 如果直接运行 `python src/train/trainer.py` 可能报错。

**Workaround**: 始终通过 `scripts/train_*.bat` 启动，或使用 `python -m src.train.trainer`。

---

## 6. 训练步数自动对齐到 n_steps

**问题**: PPO 的 `n_steps=2048`，所以 `--total-timesteps 100` 实际会跑 2048 steps。

**影响**: 短测试比预期慢。

**Workaround**: 测试时设置 `--total-timesteps 2048` 避免混淆。

---

## 7. shimmy 兼容层性能开销

**问题**: `GymV21CompatibilityV0` 在每次 step() 中做 API 转换。

**影响**: 约 5-10% 的性能损失（实测 fps=163，可接受）。

**Workaround**: 长期应升级 PantheonRL 到 gymnasium。

---

## 8. 评估时 env.reset() 的返回值格式

**问题**: 不同 gym/gymnasium 版本的 `env.reset()` 返回值不同（旧版返回 `obs`，新版返回 `(obs, info)`）。

**影响**: `evaluator.py` 中使用了 `hasattr` 检查来兼容两种格式。

**Workaround**: 代码已处理，但不够优雅。未来统一升级后可以简化。
