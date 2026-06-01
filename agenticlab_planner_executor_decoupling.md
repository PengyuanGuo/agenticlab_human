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

## 6. 当前进度评估 (2026-06-01)

整体判断：**Planner 已经基本从执行链路中独立出来，ActionSequence 中间契约也已经能从 `task_plan.json` 抽取；但 voice → planner 的入口、scene image 获取、以及 executor/backend 仍未形成可集成闭环。**

| 项 | 当前状态 | 说明 | 下一步 |
| --- | --- | --- | --- |
| Planner 独立运行 | ✅ 已跑通 | `src/agenticlab_human/planning/task_parser.py` 可独立生成 `TaskPlan`，并保存 `task_plan.json`、`action_sequence.txt`、`domain.pddl`、`problem.pddl`、`scene_image.png`。 | 把 `__main__` 从测试脚本形态收敛为稳定 CLI/API：`generate_plan(task_text, scene_image)`。 |
| PDDL + Fast Downward | ✅ 已接入 | `TaskParser.parse_task()` 在 `use_pddl=True` 时会让 VLM 生成 domain/problem，再调用 Fast Downward 生成 plan。 | 增加 planner 失败时的结构化错误返回，而不是只 fallback 到 VLM 原始 action_sequence。 |
| ActionSequence 契约 | ✅ 已建立 | `src/agenticlab_human/core/action_sequence.py` 已提供 `Action` / `ActionSequence` dataclass、`load()`、`load_from_dir()`、`from_task_plan_json()`、JSON 序列化。 | 将其视为 Planner → Executor 的唯一运行时契约，后续 executor 不直接读 `TaskPlan`。 |
| `TaskPlan` → `ActionSequence` | ✅ 已接上 | `TaskPlan.to_action_sequence()` / `to_action_sequence_dict()` 已存在，`TaskParser.save_results(..., save_action_sequence=True)` 可写 `action_sequence.json`。 | 历史 session 可能没有 `action_sequence.json`，但可以由 `ActionSequence.load(session_dir)` 从 `task_plan.json` 现算。 |
| 历史 plan 抽取验证 | ✅ 已验证 | 对 `output/task_parser/20260527_112644/task_plan.json` 抽取得到 `ActionSequence(... actions=12, goal_conditions=4)`；首个动作解析为 `pick object=green-cube-1 from=orange-cube-1`，最后动作为 `place-on-object object=blue-cube-1 target=green-cube-1`。 | 给这个样例补一个轻量单元测试，锁住 PDDL 参数名映射行为。 |
| Voice STT | ✅ 独立模块可用 | `/home/agenticlab/Project/speech_to_text_module/demo_speech_to_text.py` 当前能通过 `SpeechInputService.listen() -> str` 得到自然语言文本。 | 在 AgenticLab 侧新增 thin adapter，不要让 planner 直接 import demo 脚本。 |
| Voice → Planner | ⚠️ Adapter 已新增，待端到端验证 | `src/agenticlab_human/planning/voice_to_planner.py` 已负责 listen / 指定 image path 或 camera 取图 / 调用 planner / 保存 session。 | 用 `--scene-source camera --camera-name FemtoBolt` 跑通真实 STT → Camera → Planner。 |
| Scene image 输入 | ✅ image path 与 camera provider 已接入 | `TaskParser` 需要 `scene_image: PIL.Image`；`voice_to_planner` 现在支持静态图和 `cam_capture.CameraCapture.capture_pil()`。 | 后续按需把相机参数、曝光/分辨率等配置化。 |
| Executor/backend | ❌ 尚未实现 | `src/agenticlab_human/execution/` 目前只有 `__init__.py`，还没有 `ActionBackend`、`Action.execute_sequence()`、平台后端。 | Planner 集成 voice 之前可以先不阻塞；但端到端机器人执行前必须补。 |
| 每步 LLM verify 移除 | ⚠️ 规划侧已解耦，执行侧未落地 | 新 ActionSequence 不依赖逐步 LLM；但执行主循环还不存在，因此尚未证明“执行不调 LLM”。 | executor 验收时显式禁止 ActionChecker 默认进入主循环。 |

### 当前进度结论

- **Planning isolation：约 70% 完成。** Planner 已能单独从文本 + 图像产出持久化 plan，ActionSequence 也已经是可加载、可序列化的中间产物。
- **Voice-triggered planning readiness：约 40% 完成。** STT 和 Planner 两端都可独立工作，但中间缺一个稳定 adapter；最重要的缺口不是文本，而是 `scene_image` 获取和配置边界。
- **Executor decoupling：约 15% 完成。** 契约已准备好，但执行 API、backend protocol、无 LLM 主循环还没实现。


## 7. 阶段产出 (Deliverables)

### 已完成

1. `src/agenticlab_human/planning/task_parser.py`
   - `TaskParser.parse_task()` 可独立生成 `TaskPlan`。
   - `TaskParser.generate_plan()` 已作为一站式 planner 入口存在。
   - `save_results()` 会保存 planner 源产物，并可选保存 `action_sequence.json`。
2. `src/agenticlab_human/core/action_sequence.py`
   - `ActionSequence` / `Action` dataclass 已完成。
   - 支持从 session dir、`task_plan.json`、`action_sequence.json`、`action_sequence.txt` 加载。
   - 支持 PDDL action 参数名语义化，例如 `?obj -> object`，`?underobj -> from`，`?target -> target`。
3. Planner 独立验收样例
   - `output/task_parser/20260527_112644/task_plan.json` 已包含完整 task、objects、reasoning、PDDL、action_sequence、goal_conditions。
   - 可通过 `ActionSequence.load("output/task_parser/20260527_112644")` 抽取 executor-friendly action list。

### 下一阶段必须补齐

1. `src/agenticlab_human/planning/voice_to_planner.py`
   - 已新增 adapter：输入为 `SpeechInputService.listen()` 返回的 `task_text`，以及 `SceneProvider.capture()` 返回的 `PIL.Image`。
   - 输出：Planner session directory path + `ActionSequence`。
   - 暂时只做 planning，不碰 executor。
   - 当前 `SceneProvider` 支持指定 `image_path`，也支持通过 `CameraSceneProvider` 调用 `cam_capture.CameraCapture.capture_pil()`。
2. `src/agenticlab_human/perception/camera/cam_capture.py`
   - 作为相机模块主要对外接口。
   - 已移除 Kinect / RealSense 分支，当前只保留 Orbbec 系列入口：`Orbbec`、`FemtoBolt`、`Gemini305`。
   - 提供 `capture()` 返回 RGB-D，`capture_pil()` / `get_color_image()` 供 Planner 获取 `PIL.Image`。
3. Planner CLI
   - `python -m agenticlab_human.planning.task_parser --task "..." --image "..."`
   - `python -m agenticlab_human.planning.voice_to_planner --scene-source image --image-path "..."`
   - `python -m agenticlab_human.planning.voice_to_planner --scene-source camera --camera-name FemtoBolt`
   - CLI 只负责触发和保存，不做机器人执行。
4. ActionSequence tests
   - 用 `output/task_parser/20260527_112644/task_plan.json` 或一个裁剪版 fixture 测：
     - action 数量为 12。
     - `pick` 的第二参数映射为 `from`。
     - `place-on-object` 的第二参数映射为 `target`。
     - `goal_conditions` 被保留。
5. Executor 后续工作
   - `src/agenticlab_human/execution/action_backend.py`
   - `src/agenticlab_human/execution/action.py`
   - `python -m agenticlab_human.execution.action --plan output/task_parser/<ts>/`
   - 默认执行路径不 import / 不调用 ActionChecker。

## 8. 注意事项

- **状态检查不要全删**，但要轻量化：grasp 失败、感知未检测到目标、TCP 越界等，仍在 action 内部用确定性逻辑兜底；只有在重试用尽时才可选回退到 `ActionChecker`。
- ActionSequence 既能从 `task_plan.json` 反序列化，也能手写，便于回归测试和 ablation（绕过 Planner 直接喂 action）。
- `actionBackend` 是 Flexiv / UR / X5 / 人形复用同一套上层逻辑的关键，新增平台 = 实现一个 Backend，不动 action / Planner。

## 9. Voice 触发 Planning 的建议集成流

第一版目标不是“语音直接执行机器人”，而是：**语音输入一句自然语言任务，Planner 生成并持久化 ActionSequence。**

```
SpeechInputService.listen()
    ↓ text
VoiceTriggeredPlanner
    ↓ captures / loads scene image
TaskParser.generate_plan(task_text, scene_image)
    ↓ session_dir
ActionSequence.load(session_dir)
    ↓
action_sequence.json / task_plan.json ready for executor
```

建议第一版 API：

```python
class VoiceTriggeredPlanner:
    def __init__(self, speech_service, task_parser, scene_provider):
        self.speech_service = speech_service
        self.task_parser = task_parser
        self.scene_provider = scene_provider

    def listen_and_plan(self):
        task_text = self.speech_service.listen()
        if not task_text:
            return None
        scene_image = self.scene_provider.capture()
        task_plan = self.task_parser.generate_plan(task_text, scene_image)
        return task_plan.to_action_sequence()
```

### 集成顺序

1. **静态图验证**
   - 从 voice 得到 `task_text`。
   - 用 `StaticImageSceneProvider("/path/to/test_image.png")` 提供 scene image。
   - 生成新的 `output/task_parser/<ts>/task_plan.json` 和 `action_sequence.json`。
2. **相机图验证**
   - 使用 `--scene-source camera --camera-name FemtoBolt`，由 `cam_capture.CameraCapture` 捕获当前 RGB 图。
   - 保持 Planner 和 ActionSequence API 不变。
3. **Executor dry-run**
   - 只打印 `ActionSequence.actions`，不动机器人。
   - 验证语音任务能稳定变成结构化动作列表。
4. **Executor real-run**
   - 接入 `Action.execute_sequence(ActionSequence.load(session_dir))`。
   - 默认无 LLM verify；失败只返回结构化 error。

### 集成前置条件

- `speech_to_text_module` 应作为外部依赖或子模块被 AgenticLab adapter 调用；不要 import `demo_speech_to_text.py`，只 import `SpeechInputService`。
- `TaskParser` 的 config 路径要固定为 AgenticLab repo 内的 `configs/planning/task_parser_config.yaml`。
- voice 触发入口必须显式接收 `--scene-source image --image-path ...` 或 `--scene-source camera --camera-name ...`，因为 STT 只能提供文本，无法替代 VLM planner 的 scene image。
- planner 输出的 session dir 要返回给上层，方便 executor、调试 UI、日志系统继续读取。
