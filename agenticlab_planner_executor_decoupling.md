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
ActionExecutor.prepare()  (bbox / grasp / scene cache)
    ↓
ActionExecutor.execute_sequence()  (机器人无关的执行循环)
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
plan = planner.generate_plan(task_input, scene_image)     # 一次性产出
executor.prepare(plan.action_sequence)                    # 同步预计算 bbox / grasp cache
executor.execute_sequence(plan.action_sequence)           # 逐条执行，不调 LLM
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
| **ExecutionContext** | `ActionSequence` + RGB-D + detector/grasp backend | `SceneCache` / `GraspCache` | 否 |
| **ActionExecutor** | `ActionSequence` + `ExecutionContext` + `ActionBackend` | `ExecutionReport` | 否（默认；可选状态检查） |


> 注意：`action_checker` **不进 action 主循环**。它只作为可选的离线/事后评估器，或仅在 action 失败重试时触发，避免每步 LLM 调用。

## 5. Execution preparation / affordance cache

ActionSequence 进入 executor 后，第一版推荐先做同步 precompute，而不是立刻做多线程异步：

```text
ActionSequence
    ↓
ExecutionContext.prepare_for_sequence()
    ↓
extract interested objects / pick objects
    ↓
SceneProvider.capture_rgbd()
    ↓
PerceptionBackend.detect(rgb, object_names)
    ↓
GraspBackend.plan_scene(rgb, depth)
    ↓
assign_grasps_to_objects(all_grasps, bboxes)
    ↓
ActionExecutor.execute_sequence()
```

这样可以把 YOLO / GroundingDINO 和 AnyGrasp 放在 executor 前置准备阶段：

- detector 一次检测所有 interested objects。
- AnyGrasp 一次跑 full-scene RGB-D / point cloud，得到全场景 grasp candidates。
- 通过 bbox / mask / 3D region 把 grasp candidates 分配到各个 object。
- execution loop 中 `pick` 直接读取 cached grasp candidates，减少重复推理。

需要保留 cache invalidation，因为 world state 会随执行变化：

- `pick(object)` 后，该 object 原始 pose / bbox / grasp cache 应标记为 stale。
- `place(object, target)` 后，该 object 的新位置依赖真实执行结果，后续若还要 pick 它，需要 refresh。
- target 如果是刚被移动过的 object，place pose 不能只依赖 prepare 阶段的旧 bbox。
- 第一版可用简单策略：缓存缺失或 `context.is_stale(name)` 时，返回结构化错误或同步 refresh 单个 object；不引入 LLM verify。

## 6. Backend APIs（机器人/感知/抓取解耦）

action 写成 AgenticLab 的执行后端，而不是「能动的 demo」。底层接口统一：

```python
class ActionBackend(Protocol):
    def initialize(self) -> None: ...
    def pick(self, object_name: str, grasp_candidates=None, object_bbox=None, object_pose=None, **kwargs) -> ActionResult: ...
    def place_on_object(self, object_name: str, target_name: str, target_pose=None, target_bbox=None, **kwargs) -> ActionResult: ...
    def place_on_surface(self, object_name: str, surface_name: str, target_pose=None, **kwargs) -> ActionResult: ...
    def place_in_container(self, object_name: str, container_name: str, target_pose=None, **kwargs) -> ActionResult: ...
    def move_home(self) -> ActionResult: ...
    def get_eef_pose(self): ...
    def shutdown(self, move_home: bool = False) -> None: ...
```

感知和抓取也单独抽象，不写死在 `action.py`：

```python
class PerceptionBackend(Protocol):
    def detect(self, rgb, object_names: list[str]) -> dict[str, list[BBox]]: ...

class GraspBackend(Protocol):
    def plan_scene(self, rgb, depth) -> list[GraspCandidate]: ...
    def plan_for_object(self, rgb, depth, bbox: BBox) -> list[GraspCandidate]: ...
```

具体平台各自实现：

- `FlexivActionBackend`  ← 包装现有 `action_wrapper_flexiv.ActionWrapper`
- `UR5eActionBackend`    ← 包装现有 `action_wrapper.ActionWrapper`
- `X5ActionBackend` / `TopstarActionBackend` / `HumanoidActionBackend` … 后续按需添加
- `YoloWorldDetector` / `GroundingDinoDetector` / `MockDetector`
- `AnyGraspBackend` / `MockGraspBackend`

上层 action 调度逻辑保持平台无关：

```python
class ActionExecutor:
    def __init__(self, backend: ActionBackend, context: ExecutionContext):
        self.backend = backend
        self.context = context

    def prepare(self, action_sequence: ActionSequence) -> None:
        self.context.prepare_for_sequence(action_sequence)

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

`_do_pick` / `_do_place` 内部只读 `ExecutionContext` 的 bbox / grasp / pose cache，并显式传给 `backend.pick(...)` / `backend.place_* (...)`。`ActionExecutor` 不 import YOLO、AnyGrasp、Flexiv SDK、UR SDK，也 **不再调用 ActionChecker**。

## 7. Robot backend layer / Flexiv 单动作测试

当前 execution 的机器人实现层放在：

```text
src/agenticlab_human/execution/robot/
    flexiv/
        flexiv_controller.py
        flexiv_gripper_controller.py
        action_backend.py
```

感知 / 抓取 backend 抽象已移动到：

```text
src/agenticlab_human/perception/backend/
    perception_backend.py
    grasp_backend.py
```

Flexiv 当前测试目标不是跑完整 perception pipeline，而是验证：

```text
ActionSequence.load(...)
    ↓
ActionExecutor
    ↓
FlexivActionBackend.pick(...)
    ↓
camera-frame grasp pose
    ↓ FemtoBolt hand-eye from configs/perception/camera_config.yaml
base-frame Flexiv TCP grasp / approach pose
    ↓
execute=False: structured report only
execute=True: FlexivController + FlexivGripperController
```

测试 action 文件：

```bash
data/data_for_test/task_parser/execution/action_sequence.json
```

测试 grasp pose 目前由 CLI 注入，不经过 YOLO / AnyGrasp：

```text
translation = [0.04381273, -0.07053141, 0.536]
rotation =
[[-0.15578863,  0.90573424,  0.39417678],
 [-0.33291125, -0.423846,    0.84233284],
 [ 0.93,        0.0,         0.3675595 ]]
width = 0.08437179774045944
```

默认命令只验证 data flow，不连接、不执行机器人：

```bash
PYTHONPATH=src /usr/bin/python3 -m agenticlab_human.execution.action \
  --plan data/data_for_test/task_parser/execution/action_sequence.json \
  --backend flexiv \
  --test-grasp-camera-pose
```

真实执行必须显式加 `--execute`：

```bash
PYTHONPATH=src /usr/bin/python3 -m agenticlab_human.execution.action \
  --plan data/data_for_test/task_parser/execution/action_sequence.json \
  --backend flexiv \
  --test-grasp-camera-pose \
  --execute
```

安全约束：

- `--execute` 默认为 false。
- `FlexivActionBackend` 在 `execute=False` 时不 import `flexivrdk`，不连接 controller/gripper。
- `action.py` 只负责 ActionSequence dispatch 和 context 注入，不直接 import Flexiv SDK、YOLO、AnyGrasp、ActionChecker。
- `FlexivActionBackend.pick()` 第一版只覆盖 one-action pick 测试；place 类动作先返回结构化 not implemented error。

## 8. 当前进度评估 (2026-06-02)

整体判断：**Planner 已经基本从执行链路中独立出来，ActionSequence 中间契约也已经能从 `task_plan.json` 抽取；executor/backend 需要先落一个 dry-run + prepare/cache 骨架，再接真实 Flexiv/UR/X5。**

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
| Executor/backend | ⚠️ 正在补骨架 | `src/agenticlab_human/execution/` 需要新增 `ActionBackend`、`ActionExecutor`、`ExecutionContext`、perception/grasp backend 抽象。 | 第一版先支持 dry-run 和同步 prepare cache，不直接接真实机器人。 |
| 每步 LLM verify 移除 | ⚠️ 规划侧已解耦，执行侧未落地 | 新 ActionSequence 不依赖逐步 LLM；但执行主循环还不存在，因此尚未证明“执行不调 LLM”。 | executor 验收时显式禁止 ActionChecker 默认进入主循环。 |
| bbox + grasp 预计算 | 🆕 新增设计 | ActionSequence 出来后可一次性 detect interested objects，并 full-scene AnyGrasp 一次生成候选。 | 作为 `ExecutionContext.prepare_for_sequence()` 落地，后续支持 cache refresh。 |
| Flexiv one-action backend | 🆕 no-execute 已接通 | `FlexivActionBackend` 可读取 Flexiv/FemtoBolt config，将 camera-frame grasp 转成 base-frame TCP grasp/approach pose。 | 加 `--execute` 前先人工检查 pose6d / workspace limit / gripper width。 |

### 当前进度结论

- **Planning isolation：约 70% 完成。** Planner 已能单独从文本 + 图像产出持久化 plan，ActionSequence 也已经是可加载、可序列化的中间产物。
- **Voice-triggered planning readiness：约 40% 完成。** STT 和 Planner 两端都可独立工作，但中间缺一个稳定 adapter；最重要的缺口不是文本，而是 `scene_image` 获取和配置边界。
- **Executor decoupling：约 25% 完成。** 契约已准备好，下一步落 dry-run executor + backend/context protocol；真实机器人执行仍需平台 backend。


## 9. 阶段产出 (Deliverables)

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
   - `src/agenticlab_human/execution/execution_context.py`
   - `src/agenticlab_human/execution/perception_backend.py`
   - `src/agenticlab_human/execution/grasp_backend.py`
   - `python -m agenticlab_human.execution.action --plan output/task_parser/<ts>/`
   - 默认执行路径不 import / 不调用 ActionChecker。
   - 第一版 `prepare()` 同步预计算 bbox / grasp cache；暂不做后台异步 refresh。
6. Flexiv one-action 测试
   - `src/agenticlab_human/execution/robot/flexiv/action_backend.py`
   - `python -m agenticlab_human.execution.action --backend flexiv --test-grasp-camera-pose`
   - 默认 no-execute，只输出结构化 pick plan。

## 10. 注意事项

- **状态检查不要全删**，但要轻量化：grasp 失败、感知未检测到目标、TCP 越界等，仍在 action 内部用确定性逻辑兜底；只有在重试用尽时才可选回退到 `ActionChecker`。
- ActionSequence 既能从 `task_plan.json` 反序列化，也能手写，便于回归测试和 ablation（绕过 Planner 直接喂 action）。
- `actionBackend` 是 Flexiv / UR / X5 / 人形复用同一套上层逻辑的关键，新增平台 = 实现一个 Backend，不动 action / Planner。
- `action.py` 不直接 import YOLO / AnyGrasp；这些实现只进入 `PerceptionBackend` / `GraspBackend`。
- `action_utils.py` 只放纯函数和小型 cache helper。`move_home`、`get_eef_pose`、`convert_hand_eye` 这类平台/标定相关能力应属于 backend 或 calibration transformer，不放进通用 utils 大杂烩。
- Flexiv backend 的 `convert_hand_eye` / grasp-frame 到 tool-frame 对齐属于 robot backend 层，不放进 action loop。

## 11. Voice 触发 Planning 的建议集成流

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
    ↓
ActionExecutor.prepare(action_sequence)
    ↓
ActionExecutor.execute_sequence(action_sequence)
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
   - 接入 `ActionExecutor.execute_sequence(ActionSequence.load(session_dir))`。
   - 默认无 LLM verify；失败只返回结构化 error。

### 集成前置条件

- `speech_to_text_module` 应作为外部依赖或子模块被 AgenticLab adapter 调用；不要 import `demo_speech_to_text.py`，只 import `SpeechInputService`。
- `TaskParser` 的 config 路径要固定为 AgenticLab repo 内的 `configs/planning/task_parser_config.yaml`。
- voice 触发入口必须显式接收 `--scene-source image --image-path ...` 或 `--scene-source camera --camera-name ...`，因为 STT 只能提供文本，无法替代 VLM planner 的 scene image。
- planner 输出的 session dir 要返回给上层，方便 executor、调试 UI、日志系统继续读取。
