# Group A Baseline — 进度追踪

> 根据 `group_a_implementation_spec.md` 检查 CC 已完成的工作和剩余工作。

---

## 当前进度总览

| Phase | 内容 | 状态 | 备注 |
|-------|------|------|------|
| Phase 1 | Skeleton + Protocol | ✅ **完成** | protocol.py, test_protocol.py — 8/8 测试通过 |
| Phase 2 | Wrapper | ✅ **完成** | human_feedback_wrapper.py, factory.py, test_wrapper.py — 8/8 测试通过 |
| Phase 3 | Feedback Sources | ✅ **完成** | keyboard_listener.py, random_source.py, null_source.py |
| Phase 4 | Trainer + Callbacks | ✅ **完成** | trainer.py 已成功运行 2048 steps (15.5s)，生成了 final_model.zip |
| Phase 5 | Evaluator + Reproducibility | ✅ **完成** | evaluator.py, test_reproducibility.py — 4/4 测试通过 |
| Phase 6 | 文档 + Demo 脚本 | ✅ **完成** | architecture.md, design_decisions.md, known_issues.md, how_to_demo.md |

---

## 已验证的功能

### ✅ Phase 1-3 单元测试
- `pytest tests/test_protocol.py` — **8/8 通过**
- `pytest tests/test_wrapper.py` — **8/8 通过**

### ✅ Phase 4 端到端训练
- `python scripts/test_trainer.py` — **成功运行**
  - 使用 CUDA GPU (RTX 5060)
  - 2048 timesteps (PPO n_steps=2048)
  - 生成文件: `final_model.zip`, `monitor.monitor.csv`, `trajectories.jsonl`, `tb/`, `train_summary.json`

### ⚠️ 已知问题
1. **gym 0.25 兼容性警告** — 不影响功能，但需要升级到 gymnasium
2. **PPO GPU 警告** — MlpPolicy 建议用 CPU，但 GPU 也能跑
3. **Shell 脚本是 bash 格式** — Windows 需要 `.bat` 格式
4. **`factory.py` 中的 `shimmy` 兼容层** — 工作正常但增加了复杂度

---

## 剩余工作清单

### Phase 5: Evaluator + Reproducibility ✅
- [x] 创建 `src/eval/evaluator.py` — 加载 checkpoint 跑 N 个 evaluation episode
- [x] 创建 `scripts/evaluate.bat` — 评估启动脚本
- [x] 创建 `tests/test_reproducibility.py` — 同 seed 结果一致性测试 — 4/4 通过
- [x] 创建 `src/utils/repro.py` — seed 设置工具函数
- [x] 创建 `src/utils/ipc.py` — 进程间通信辅助

### Phase 6: 文档 + Demo 脚本 ✅
- [x] 创建 `docs/architecture.md` — 架构图 + 数据流
- [x] 创建 `docs/design_decisions.md` — 关键设计决策（8 条 D-XXX）
- [x] 创建 `docs/known_issues.md` — 已知问题 + workaround
- [x] 创建 `docs/how_to_demo.md` — 周三组会 demo 脚本
- [ ] 完善顶层 `README.md`（可选）

### Windows 适配 ✅
- [x] 创建 `scripts/train_no_feedback.bat`
- [x] 创建 `scripts/train_random_feedback.bat`
- [x] 创建 `scripts/train_human_feedback.bat`

### 最终验收 ✅
- [x] `pytest tests/` 全绿 — **16/16 通过**
- [x] no-feedback mode 已验证 — **2048 steps 成功运行**
- [ ] random-feedback mode 待验证
- [ ] human-feedback mode 待验证（需要显示器）
- [ ] git commit 并 push
