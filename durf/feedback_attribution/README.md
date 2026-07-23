# Feedback Attribution

这个包是新版研究方向的第一层代码骨架：

```text
自然语言反馈 + 游戏轨迹事实
-> 候选事件
-> 语义归因
-> H_u 训练样本
```

目前这里的大部分脚本都是离线脚本。也就是说，它们先读取一局游戏保存下来的日志，再做转换、事件检测和归因预览。

## 当前最小流程

先正常运行一局游戏：

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.group_a.play_with_baseline --layout cramped_room
```

然后对这局游戏的 session 文件夹运行离线归因 demo：

```powershell
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id>
```

如果要启用 DeepSeek 语义归因，先设置 API key，然后加 `--use-llm`：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --use-llm
```

它会写出：

```text
trajectory.jsonl
feedback_events.jsonl
candidate_events.jsonl
attribution_preview.jsonl
llm_attribution_audit.jsonl  # only when --use-llm is enabled
```

## 文件职责说明

- `schemas.py`
  - 它是“数据格式定义文件”。
  - 回答的问题是：这个项目里一条轨迹、一条反馈、一个候选事件、一个归因结果，应该长什么样？
  - 现在里面定义了四种核心记录：
    - `trajectory_step`：一帧/一步游戏轨迹。
    - `feedback_event`：一次人类反馈，目前主要来自聊天语言反馈；旧 session 里的 `+1/-1` 只做兼容读取。
    - `candidate_event`：程序从轨迹里检测出来的候选协作事件。
    - `attribution_result`：把人类反馈归因到某个时间窗口/事件后的预览结果。
  - 为什么需要它：如果每个脚本随手写字段名，后面会乱成一团；这个文件负责统一字段口径。

- `io_utils.py`
  - 它是“文件读写小工具箱”。
  - 回答的问题是：怎么稳定地读 CSV、写 JSONL、读 JSONL，以及把 CSV 里的字符串转成数字、布尔值或 JSON？
  - 现在里面有：
    - `read_csv`：读取 `.csv` 表格。
    - `write_jsonl`：把一组字典逐行写成 `.jsonl`。
    - `read_jsonl`：读取 `.jsonl`。
    - `as_int` / `as_float` / `as_bool` / `as_json`：把 CSV 字符串转换成程序真正需要的类型。
  - 它不包含研究逻辑，只负责让其他脚本少写重复代码。

- `session_converter.py`
  - 它是“原始游戏日志转换器”。
  - 回答的问题是：Pygame 游戏跑完后产生的原始 CSV，怎么变成后续归因流程更好处理的 JSONL？
  - 输入文件：
    - `trajectory.csv`：每一步游戏动作、奖励、状态快照。
    - `chat_messages.csv`：人类在 Chat 里输入的语言反馈，以及模型回复。
    - `feedback.csv`：旧版本的 `J/K +1/-1` 标量反馈，只为了兼容旧 session，新 session 不再生成。
  - 输出文件：
    - `trajectory.jsonl`：结构化后的每一步轨迹。
    - `feedback_events.jsonl`：结构化后的人类反馈事件。
  - 关键点：新版本 `trajectory.csv` 里的 `state_after_json` 会被放进 `trajectory.jsonl` 的 `state_facts` 里，里面包括 AI/人类位置、手持物、锅状态、地图结构等。

- `event_detectors.py`
  - 它是“事件识别规则库”。
  - 回答的问题是：只看游戏轨迹，不看人类说了什么，程序能不能先找出一些可能值得评价的协作事件？
  - 它不调用 LLM，也不判断“用户到底想表达什么”。它只做事实层面的候选事件检测。
  - 当前检测器覆盖四组事实：
    - 通行与协作问题，例如 `AI_blocked_human_path`。
    - 错失任务机会，例如忽略锅、未准备盘子/原料、错过 counter 物体和劳动分工。
    - 明显异常模式，例如反复拿放、长期拿着不需要的物体。
    - 正向任务进展，例如成功放原料、取汤和送餐。
  - 完整分类、触发条件和已知误报边界见 `docs/candidate_event_taxonomy.md`。
  - 输出的是 `candidate_event`，里面会包含：
    - 事件类型
    - 起止 timestep
    - 证据字段，比如位置、动作、锅状态
    - confidence
    - severity
    - actor
    - event_valence

- `condition_features.py`
  - 它是“条件事实提取器”。
  - 它从每一步状态中计算固定 Hu 布尔条件，并保存 Review 所需的原始上下文。
  - 当前包括：
    - 谁拿着什么、锅是否缺某种原料。
    - 人类是否拿着最后所需原料。
    - 双方谁更靠近盘子、锅或所需原料。
    - counter 是否已有可用物体。
    - `AI -> source -> pot` 的总路径成本，以及物体是否已被 staging 在锅边。
  - Hu-v0 只编码固定布尔条件；物体类型和原始距离只用于溯源与 Review。

- `generate_candidate_events.py`
  - 它是“候选事件生成命令”。
  - 回答的问题是：给我一个完整 session 文件夹，怎么一键生成 `candidate_events.jsonl`？
  - 它会先确保 session 被转换成 JSONL，然后调用 `event_detectors.py` 里的检测器。
  - 输入是一个 session 文件夹，例如：
    - `outputs/human_ai_sessions/20260624_225407`
  - 输出是：
    - `candidate_events.jsonl`
  - 常用命令：
    ```powershell
    python -m durf.feedback_attribution.generate_candidate_events `
      --session outputs\human_ai_sessions\<session_id>
    ```

- `feedback_type_router.py`
  - 它是“反馈类型粗分类器”。
  - 回答的问题是：一句反馈大概属于哪类？
  - 目前是非常初级的规则版本，用关键词粗略区分：
    - evaluative：评价式反馈，比如“刚刚很好”“不对”“bad”。
    - imperative：指令式反馈，比如“去拿盘子”“你应该去锅那边”。
    - descriptive：描述式反馈，比如“你堵住我了”“你一直重复做这个”。
  - 它还会粗略判断 polarity：
    - positive
    - negative
    - neutral
  - 这个文件现在更像 baseline/占位符。之后真正重要的语义归因会交给 LLM，但这个文件可以作为非 LLM 对照的一部分。

- `sample_builder.py`
  - 它是“归因预览构造器”。
  - 回答的问题是：有了一条人类反馈和已经检测出的候选事件后，能不能先生成一个保守的 attribution preview？
  - 现在它做的是早期预览：
    - 调用反馈类型分类器。
    - 找反馈发生时间附近的 `candidate_events`。
    - 用简单关键词把语言反馈对齐到候选事件。
      - 例如“别堵我”优先对齐到 `AI_blocked_human_path`。
      - 例如“为什么不拿汤”优先对齐到 `AI_ignored_ready_or_nearly_ready_pot`。
    - 生成一条 `attribution_result`。
    - 针对反馈时刻重新检测近期窗口，保证不会读取反馈之后的未来轨迹。
  - 当前结果不是最终训练 `H_u` 的数据。它的作用是帮我们检查流程有没有打通、字段够不够、哪里需要澄清。
  - 它是 LLM 语义归因之前的确定性 baseline。

- `llm_attributor.py`
  - 它是“LLM 语义归因器”。
  - 回答的问题是：当程序已经检测出若干候选事件后，人类这句话到底更可能指向哪个事件？
  - 它不会自己检测底层事实，也不会改写环境奖励函数。
  - 它收到的输入包括：
    - 一条人类反馈，比如“你为什么不拿汤”。
    - 附近的 `candidate_events`。
    - 最近几步轨迹摘要。
    - 规则 baseline 的初步判断。
  - 它的职责是：
    - 从候选事件中选择 `target_event`。
    - 判断反馈类型和情绪倾向。
    - 提取关键条件 `key_conditions`。
    - 给出置信度 `confidence`。
    - 如果语义不清，设置 `needs_clarification=true` 并提出一个澄清问题。
    - 如果现有候选事件无法表达用户意思，写入 `proposed_schema_update`，供之后人工审核。
  - 它必须返回 JSON。脚本会把 prompt、raw response、parsed response 写入 `llm_attribution_audit.jsonl`，保证之后可以复查。
  - 如果 DeepSeek 没有配置、调用失败或输出不合法，系统会自动回退到 `sample_builder.py` 的规则 baseline。

- `demo_offline_attribution.py`
  - 它是“完整离线流程的一键 demo”。
  - 回答的问题是：一个 session 能不能从原始 CSV 一路跑到 attribution preview？
  - 它串起了现在的最小流程：
    ```text
    trajectory.csv / chat_messages.csv
    -> trajectory.jsonl / feedback_events.jsonl
    -> candidate_events.jsonl
    -> attribution_preview.jsonl
    ```
  - 常用命令：
    ```powershell
    python -m durf.feedback_attribution.demo_offline_attribution `
      --session outputs\human_ai_sessions\<session_id>
    ```
  - 它适合用来快速检查一个新 session 是否能通过整个离线管线。
  - 加上 `--use-llm` 后，它会调用 `llm_attributor.py`，并额外写出 `llm_attribution_audit.jsonl`。

- `__init__.py`
  - 它是 Python 包标记文件。
  - 作用是告诉 Python：`feedback_attribution` 是一个可以被导入和用 `python -m ...` 运行的包。
  - 没有它，类似下面的命令可能无法正常工作：
    ```powershell
    python -m durf.feedback_attribution.demo_offline_attribution
    ```

如果只想生成候选事件，运行：

```powershell
python -m durf.feedback_attribution.generate_candidate_events `
  --session outputs\human_ai_sessions\<session_id>
```

第一版事件检测器故意做得很小、很保守。默认归因窗口现在是
`25` 个 timestep，也就是在当前 0.5 秒一步的设置下约 12.5 秒：

- `AI_blocked_human_path`：人类尝试移动到 AI 占据的格子，但移动失败。
- `AI_ignored_ready_or_nearly_ready_pot`：锅已经 ready，AI 有可能处理锅，但没有及时处理。
- `AI_failed_to_prepare_ingredient_while_waiting`：锅正在 cooking、人类拿着盘子等待时，AI 空手但没有去准备下一份原料。
- `AI_missed_useful_counter_object`：存在同类 staged counter 物体，但 AI 仍从 dispenser 取物。
- `AI_missed_labor_division_opportunity`：队友已覆盖最后原料或盘子角色时，AI 没有选择互补任务。

## 当前限制

新版 `durf.group_a.play_with_baseline` 生成的 session 已经会在 `trajectory.csv`
里记录 `state_before_json` 和 `state_after_json`。这些字段包括：

- AI 和人类的位置
- AI 和人类手里拿着什么
- 锅状态
- 地图上的物体
- 地图 terrain 结构

旧 session 可能没有这些字段。对于旧 session，`attribution_preview` 不能当作
训练数据，只能用来检查文件流程是否打通，以及暴露缺少哪些状态事实。

`J/K +1/-1` 标量反馈现在只属于旧版本兼容逻辑。当前研究路线使用
`chat_messages.csv` 里的自然语言反馈；`feedback.csv` 只在读取旧 session 时兼容使用。
