# AgenticLab Planner / Action 解耦设计

## 1. 目标

将「规划」和「动作执行」拆成两个独立模块，去掉每步依赖 LLM 的闭环 verify，让上层架构变成单向数据流：

```
User / Task Input
    ↓
Planner (LLM / VLM / PDDL)
    ↓
ActionSequence  (标准化中间产物，文件可持久化)
    ↓
Action (机器人无关的执行后端)
    ↓
Robot Controller (Flexiv / UR / X5 / 小拓 / 人形 …)
```

中间状态检查（可选）只在必要节点触发，**不再每一步调用 LLM**。

## 2. 旧 vs 新流程

**旧流程**：在 `stacking_pipeline_flexiv.py` 里规划、感知、执行、ActionChecker 全部缠在一个 `while` 循环里：

```python
while task_not_done:
    observe()
    call_llm()           # task_parser
    execute_one_action() # action_wrapper.pick / place
    verify()             # action_checker (precondition / effect / completion)
```

**新流程**：两段，明确解耦：

```python
plan = planner.generate_plan(task_input, scene_image)   # 一次性产出
action.execute_sequence(plan.action_sequence)         # 逐条执行，不调 LLM
```

## 3. 标准 ActionSequence 格式

直接复用 `output/task_parser/<ts>/action_sequence.txt` 中的 PDDL-style 字符串，**外加 JSON schema** 作为 action 的输入契约：

```json
{
  "task": "stack_numbered_blocks_on_yellow_bin",
  "task_description": "Place the three blocks on the yellow bin ...",
  "actions": [
    { "id": 1, "name": "pick",             "args": { "object": "numbered-block-1", "from":   "gray-cloth-1" } },
    { "id": 2, "name": "place-on-object",  "args": { "object": "numbered-block-1", "target": "yellow-bin-1" } },
    { "id": 3, "name": "pick",             "args": { "object": "numbered-block-2", "from":   "gray-cloth-1" } },
    { "id": 4, "name": "place-on-object",  "args": { "object": "numbered-block-2", "target": "numbered-block-1" } }
  ]
}
```

- 来源：`TaskParser.parse_task()` 已经输出 `task_plan.json`（含 `action_sequence`、`action_details`、`goal_conditions`），只需新增一个 `to_action_sequence_json()` 适配器。
- 持久化：与 `action_sequence.txt`、`domain.pddl`、`problem.pddl` 同目录，action 可直接从磁盘加载，**Planner 和 action 之间没有内存耦合**。

## 4. 模块边界


| 模块                 | 输入                                 | 输出                          | 是否依赖 LLM               |
| ------------------ | ---------------------------------- | --------------------------- | ---------------------- |
| **Planner**        | `task_description` + `scene_image` | `ActionSequence` (json/txt) | 是（VLM + Fast Downward） |
| **ActionSequence** | —                                  | dataclass / json            | 否                      |
| **action**         | `ActionSequence` + 实时感知            | `ExecutionResult`           | 否（默认；可选状态检查）           |


> 注意：`action_checker` **不进 action 主循环**。它只作为可选的离线/事后评估器，或仅在 action 失败重试时触发，避免每步 LLM 调用。

## 5. Action Backend API（机器人无关）

action 写成 AgenticLab 的执行后端，而不是「能动的 demo」。底层接口统一：

```python
class ActionBackend(Protocol):
    def execute_pick(self, object_name: str, **kwargs) -> ActionResult: ...
    def execute_place(self, target_name: str, **kwargs) -> ActionResult: ...
    def execute_move_to_pose(self, pose: np.ndarray, frame: str = "base") -> ActionResult: ...
    def execute_open_gripper(self, width: float | None = None) -> ActionResult: ...
    def execute_close_gripper(self) -> ActionResult: ...
    def get_eef_pose(self) -> np.ndarray: ...
    def reset(self) -> None: ...
    def shutdown(self, move_home: bool = False) -> None: ...
```

具体平台各自实现：

- `FlexivactionBackend`  ← 包装现有 `action_wrapper_flexiv.ActionWrapper`
- `UR5eactionBackend`    ← 包装现有 `action_wrapper.ActionWrapper`
- `X5actionBackend` / `TopstaractionBackend` / `HumanoidactionBackend` … 后续按需添加

上层 action 调度逻辑保持平台无关：

```python
class Action:
    def __init__(self, backend: actionBackend, perception):
        self.backend = backend
        self.perception = perception  # obj_detector + grasp_planner + place_planner

    def execute_sequence(self, action_sequence: ActionSequence) -> ExecutionReport:
        for action in action_sequence.actions:
            result = self.execute(action)
            if not result.success:
                self.handle_failure(action, result)
                break
        return ExecutionReport(...)

    def execute(self, action) -> ActionResult:
        match action.name:
            case "pick":            return self._do_pick(action.args)
            case "place-on-object": return self._do_place(action.args)
            case _: raise NotImplementedError(action.name)
```

`_do_pick` / `_do_place` 内部只调感知 (`ObjectDetector`、`GraspPlanner`、`PlacePlanner`) + `backend.execute_*`，**不再调用 ActionChecker**。

## 6. 当前已有 vs 待补齐


| 项                               | 现状                                                             | 待办                                                                                                                       |
| ------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Planner（含 PDDL + Fast Downward） | ✅ `vlm_robobench/modules/planning/task_parser.py`              | 加 `load_from_dir(session_dir)` / `to_action_sequence_json()`                                                             |
| ActionSequence 数据结构             | ⚠️ 现有 `TaskPlan.action_sequence` 是字符串列表                        | 新增 `ActionSequence` dataclass（`task`, `actions[{id,name,args}]`），与现有 PDDL 字符串互转                                          |
| 机器人接口                           | ✅ `action_wrapper.py`、`action_wrapper_flexiv.py`               | 抽 `actionBackend` Protocol；两个 wrapper 仅暴露 `execute_*` 语义                                                                 |
| action 主循环                      | ❌ 当前混在 `*_pipeline_flexiv.py` 里                                | 新增 `vlm_robobench/modules/execution/action.py`，把 `execute_pick` / `execute_place` 从 pipeline 抽出去，**移除 ActionChecker 调用** |
| 文本输入 → 动作序列                     | ✅ `python -m vlm_robobench.modules.planning.task_parser` 可独立运行 | —                                                                                                                        |
| 不依赖 LLM 逐步执行                    | ❌                                                              | 新增 `python -m vlm_robobench.modules.execution.action --plan output/task_parser/<ts>/`                                    |


## 7. 阶段产出 (Deliverables)

1. `vlm_robobench/modules/execution/action_sequence.py` —— `ActionSequence` / `Action` dataclass + JSON 序列化、从 `task_plan.json` 加载。
2. `vlm_robobench/modules/execution/action_backend.py` —— `actionBackend` Protocol；`FlexivactionBackend`、`UR5eactionBackend` 适配器（薄包装现有 `ActionWrapper`）。
3. `vlm_robobench/modules/execution/action.py` —— 平台无关 `action`，`execute_sequence()` 主循环。
4. 重构 `test/agent_test/stacking_pipeline_flexiv.py`：拆成 `plan.py` + `run.py`，`run.py` 直接读取 `output/task_parser/20260520_173836/action_sequence.txt`，**不再 import `ActionChecker`**。
5. 验收：
  - 文本输入 → 生成 `action_sequence.json/.txt`（Planner 独立可跑）。
  - 给定一份历史 `action_sequence.txt`，action 可在不调用任何 LLM 的前提下跑完整段序列（action 独立可跑）。

## 8. 注意事项

- **状态检查不要全删**，但要轻量化：grasp 失败、感知未检测到目标、TCP 越界等，仍在 action 内部用确定性逻辑兜底；只有在重试用尽时才可选回退到 `ActionChecker`。
- ActionSequence 既能从 `task_plan.json` 反序列化，也能手写，便于回归测试和 ablation（绕过 Planner 直接喂 action）。
- `actionBackend` 是 Flexiv / UR / X5 / 人形复用同一套上层逻辑的关键，新增平台 = 实现一个 Backend，不动 action / Planner。

