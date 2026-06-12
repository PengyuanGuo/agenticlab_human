# 无规划执行流水线

## 目标

把现有的 X5 RGB-D 相机、fine-tuned YOLO、GraspNet HTTP 服务和 X5 HTTP
控制整合为一条可以直接运行的执行流水线。

第一阶段只支持：

```text
pick(object) -> place(object, target)
```

对象名和目标名由函数参数或 CLI 显式提供。该流程不调用：

- TaskParser；
- PDDL planner；
- LLM 或 VLM；
- ActionChecker；
- task description。

后续规划模块仍然可以生成相同的 `ActionSequence`，但不参与本阶段的
perception-to-execution 流程。

## 核心设计结论

### 不建立第二套 Executor

继续使用现有执行主链：

```text
Action
  -> ExecutionContext
  -> ActionExecutor
  -> ActionBackend
  -> RemoteX5ActionBackend
  -> X5HTTPClient
  -> X5 hardware server
```

新代码只增加一个较薄的 `ExecutionRuntime`，负责：

```text
显式 object/target
  -> 获取对齐的 RGB-D
  -> fine-tuned YOLO 检测
  -> 计算 grasp 或 place target
  -> 填充 ExecutionContext
  -> 调用 ActionExecutor.execute_action()
```

`ExecutionRuntime` 是有状态对象，因为 X5 的 place 依赖前一次 pick 保存的：

- pick trajectory；
- grasp retreat pose；
- 当前是否持有物体；
- X5 HTTP client 和 backend 生命周期。

检测选择、深度处理、坐标反投影和矩阵变换仍然使用独立函数。

runtime 在一次 manipulation session 开始时初始化 backend，在 pick 和 place
之间保持同一个 backend 实例，结束时再 shutdown。

当前不直接使用 `ActionExecutor.execute_sequence()` 执行这条无规划流水线。
原因是 place 必须在 pick 后重新拍摄并更新 context，同时保留 backend 中的
pick 状态。runtime 应在同一个 backend 生命周期内分别调用两次
`ActionExecutor.execute_action()`。

## 强制命名约定

### 坐标变换统一使用 `T_target_source`

代码中的坐标变换变量只使用：

```text
T_target_source
```

含义：

- `T_target_source` 表示 source 坐标系在 target 坐标系中的位姿；
- 它把 source 坐标系表达的点转换为 target 坐标系表达；
- 计算形式为 `p_target = T_target_source @ p_source`。

标准变量名：

```text
T_camera_grasp
T_grasp_tcp
T_world_camera
T_world_tcp
```

禁止在代码中为同一个变换混用以下别名：

```text
world_from_camera
camera_to_world
camera_from_grasp
camera_to_grasp
```

例如只写：

```python
# 相机坐标系在世界坐标系中的位姿；把 camera-frame point 转为 world-frame point。
p_world = T_world_camera @ p_camera_h
```

注释可以使用自然语言解释 frame 含义，但变量名仍然必须保持
`T_target_source`。

矩阵组合也按相同规则书写：

```text
T_world_tcp =
    T_world_camera
    @ T_camera_grasp
    @ T_grasp_tcp
```

实现和 code review 时不接受同一文件同时出现 `T_world_camera`、
`world_from_camera` 和 `camera_to_world`。

### 检测器统一表示 Fine-tuned YOLO

当前执行方向是固定类别的 fine-tuned YOLO，不是开放词汇 YOLO-World。

现有文件：

```text
perception/detection/yolo_world_detector.py
```

及类名：

```text
YoloWorldDetector
```

能够加载 regular YOLO checkpoint，但名称会误导 pipeline 使用者。执行流水线
的公开命名统一为：

```text
FineTunedYoloDetector
```

实现阶段可以选择：

1. 将现有实现重命名为 `fine_tuned_yolo_detector.py` 和
  `FineTunedYoloDetector`；
2. 暂时增加一个薄 adapter，但 pipeline、配置和日志中不再暴露
  `YoloWorldDetector`。

标准配置必须显式指定训练后的 checkpoint：

```yaml
detector:
  type: fine_tuned_yolo
  model_path: data/checkpoints/<fine-tuned-model>.pt
  confidence: 0.25
  image_size: 960
```

不再需要 `detector_model_type: regular`，因为 detector 类型已经明确。
真机执行不得静默回退到通用 YOLO 或 YOLO-World checkpoint。

### X5 Place Offset 只使用 World-X

当前 X5 的最终放置位置补偿只沿 world-X 轴，因此配置使用一个标量：

```yaml
place_offset_world_x_m: null
```

计算方式：

```python
p_world_place = p_world_target.copy()
p_world_place[0] += place_offset_world_x_m
```

不要在 X5 pipeline 中使用容易误解的三维配置：

```text
place_offset_world_m: [x, y, z]
```

`place_offset_world_x_m` 是目标放置点的标定补偿。

它和现有 backend 中的 `place_approach_offset_x_m` 含义不同：

- `place_offset_world_x_m`：修正最终 place target；
- `place_approach_offset_x_m`：由最终 place pose 生成 preplace pose。

两者都沿 world-X，但不能合并成同一个配置项。

## 现有模块

### 已具备


| 能力                | 现有模块                                                         | 当前接口                             |
| ----------------- | ------------------------------------------------------------ | -------------------------------- |
| X5 RGB-D 取图       | `execution.robot.x5.client.X5HTTPClient`                     | `capture_rgbd() -> RGBDFrame`    |
| X5 运动和夹爪          | `execution.robot.x5.x5_remote_backend.RemoteX5ActionBackend` | `pick(...)`, `place(...)`        |
| 语义动作执行            | `execution.action.ActionExecutor`                            | 执行 `pick` 和 `place`              |
| 执行期缓存             | `execution.execution_context.ExecutionContext`               | bbox、grasp、object pose           |
| YOLO 推理基础         | `perception.detection.yolo_world_detector`                   | 支持加载 regular YOLO checkpoint，待更名 |
| Grasp HTTP client | `perception.grasping.client.GraspNetHTTPClient`              | 基于 bbox 和文件路径请求                  |
| Grasp HTTP server | `perception.grasping.server`                                 | 返回 camera-frame grasp pose       |
| 手眼标定              | `configs/perception/camera_config.yaml`                      | Gemini335 的 `T_world_camera`     |


### 尚未完成的连接

1. fine-tuned YOLO 还没有清晰的 pipeline-facing 名称和 adapter。
2. `GraspNetHTTPClient` 外面还没有 `GraspBackend` adapter。
3. `ExecutionContext.prepare_for_sequence()` 偏向一次 full-scene grasp，
  HTTP GraspNet 更适合针对单个 bbox 调用 `plan_for_object()`。
4. place target 的 bbox 还没有转换为 world-frame XYZ。
5. 还没有一个保持 pick/place 状态的公开 runtime。

## 标准架构

```text
本地执行进程
├── ExecutionRuntime
│   ├── X5HTTPClient + capture_scene_from_x5_server()
│   ├── FineTunedYoloDetector
│   ├── GraspNetHTTPBackend
│   ├── place target 几何函数
│   ├── ExecutionContext
│   ├── ActionExecutor
│   └── RemoteX5ActionBackend
│
├── HTTP -> X5 server :8000
│   ├── Orbbec 对齐 RGB-D
│   ├── X5 arm commands
│   └── 单夹爪 command
│
└── HTTP -> GraspNet server :8010
    └── camera-frame grasp candidates
```

当前 GraspNet API 传递的是 RGB-D 文件路径，而不是上传数组。因此执行进程和
GraspNet server 必须能够访问同一个 capture 目录。

## 标准数据结构

### `SceneSnapshot`

`SceneProvider` 不是第一阶段功能闭环的必要模块。现有
`X5HTTPClient.capture_rgbd()` 和 `save_rgbd_frame()` 已经完成取图与保存，
不为架构完整重复实现。

第一阶段只增加 `SceneSnapshot`，统一传递 frame 和文件路径：

```python
@dataclass(frozen=True)
class SceneSnapshot:
    frame: RGBDFrame
    rgb_path: Path
    depth_path: Path
    metadata_path: Path
```

采集函数保持直接：

```python
def capture_scene_from_x5_server(
    client: X5HTTPClient,
    save_dir: Path,
) -> SceneSnapshot:
    frame = client.capture_rgbd()
    paths = save_rgbd_frame(frame, save_dir)
    return SceneSnapshot(
        frame=frame,
        rgb_path=paths["rgb"],
        depth_path=paths["depth_npy"],
        metadata_path=paths["metadata"],
    )
```

pick 和 place 分别调用一次该函数。detector 使用 `snapshot.frame.rgb`，
place estimator 使用 `snapshot.frame.depth_mm` 和
`snapshot.frame.intrinsics`，GraspNet 使用 snapshot 中保存的路径。

等出现本地回放、mock camera、ROS 或其他相机来源时，再抽象：

```text
SceneProvider.capture() -> SceneSnapshot
```

原则是：先统一数据，不提前统一数据来源。

### Detection

继续使用现有 `DetectionResult` 和 `BBox`。

最小流程：

1. fine-tuned YOLO 只检测请求的固定类别；
2. 多个同类目标时，默认选择 confidence 最高的 bbox；
3. place 时对选中结果调用：

```python
detection.set_random_obj_point(margin_ratio=0.5)
```

1. 将随机点作为 place 初始像素；
2. 找不到目标时在机器人运动前失败。

### Grasp

继续使用现有 `GraspCandidate`：

```text
pose: T_camera_grasp, 4x4
score
image_xy
object_name
metadata.frame = "camera"
metadata.width
```

GraspNet adapter 将选中的 object bbox 作为 workspace mask，并把 HTTP response
转换为 `GraspCandidate`。

### Place Target

第一阶段 place 输入为：

```text
target_pose = world-frame XYZ
```

暂不区分：

- place-on-object；
- place-on-surface；
- place-in-container。

对 executor 来说它们统一为：

```text
place(object, target)
```

## `execute_place` 标准实现

### 输入

```text
runtime
object_name
target_name
```

前置条件：

- 当前 runtime 已成功执行 pick；
- `held_object == object_name`；
- backend 仍保存 pick retreat pose；
- X5 client 和 backend 没有在 pick 后 shutdown。

### 第一步：重新获取 RGB-D

```python
frame = runtime.x5_client.capture_rgbd()
rgb_image = Image.fromarray(frame.rgb)
```

place 必须重新拍摄。pick 已经改变场景，不能继续使用 pick 前的目标位置。

### 第二步：检测目标并选择随机放置像素

```python
detection = runtime.detector.detect(rgb_image, [target_name])
if not detection.success or detection.num_objects == 0:
    return failed_place("Target detection failed")

detection.set_random_obj_point(margin_ratio=0.5)
target_pixel = detection.get_object_center()
```

随机点必须落在 bbox 内部的安全区域。检测结果和随机点都需要保存，便于复现。

### 第三步：Pixel 转 Camera XYZ

`plan_placing_point()` 保持为一个直接的几何函数：

```python
placing_result = plan_placing_point(
    rgb=rgb_image,
    depth_mm=frame.depth_mm,
    initial_pixel=target_pixel,
    target_name=target_name,
    intrinsics=frame.intrinsics,
)
```

不再传递 `task_description`。

也不需要通过 `which_camera` 再查一次内参，因为当前 HTTP frame 已携带本次
capture 的 `CameraIntrinsics`。`camera_name` 只用于从配置选择对应的
`T_world_camera`。

像素反投影：

```text
z = depth_mm / 1000
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

随机点可能正好位于深度空洞，因此不能只读取一个 depth pixel。标准做法是：

1. 以随机点为中心取一个小 patch；
2. 删除零值、负值和非有限值；
3. 计算有效 depth 的中值；
4. 没有有效 depth 时立即失败。

输出：

```text
p_camera = [x, y, z, 1]
```

### 第四步：Camera XYZ 转 World XYZ

只使用标准命名：

```python
# T_world_camera 表示相机坐标系在世界坐标系中的位姿。
p_world_h = T_world_camera @ p_camera
p_world_target = p_world_h[:3]
```

然后应用 X5 world-X 标定补偿：

```python
p_world_place = p_world_target.copy()
p_world_place[0] += place_offset_world_x_m
```

保存：

- RGB 和 depth；
- detector 输出；
- target bbox；
- 随机 place pixel；
- patch 内有效 depth；
- `p_camera`；
- `T_world_camera`；
- `p_world_target`；
- `place_offset_world_x_m`；
- `p_world_place`。

### 第五步：使用固定 Place Orientation 执行

X5 place 使用配置中的固定姿态：

```yaml
default_place_orientation_rotvec: [rx, ry, rz]
```

最终 pose：

```python
place_pose_xyz_rotvec = np.concatenate(
    [p_world_place, default_place_orientation_rotvec]
)
```

不要使用输入 bbox 或 place planner 产生的 rotation。第一阶段统一使用
X5 标定过的 `default_place_orientation_rotvec`。

将目标写入 context，再通过标准 executor 调用：

```python
runtime.context.bboxes[target_name] = selected_bboxes
runtime.context.object_states.setdefault(target_name, {})["pose"] = p_world_place

action = Action(
    id=runtime.next_action_id(),
    name="place",
    args={"object": object_name, "target": target_name},
)
result = runtime.executor.execute_action(action)
```

`RemoteX5ActionBackend.place()` 根据 world XYZ 和固定 orientation 生成运动。

标准运动顺序：

```text
从 grasp pose retreat
  -> home
  -> preplace
  -> linear move to place
  -> open gripper
  -> release retreat
  -> 可选 home
```

其中：

```text
preplace.x = place.x + place_approach_offset_x_m
```

### 参考伪代码

```python
def execute_place(
    runtime: ExecutionRuntime,
    object_name: str,
    target_name: str,
) -> ActionResult:
    runtime.require_held_object(object_name)

    frame = runtime.x5_client.capture_rgbd()
    rgb_image = Image.fromarray(frame.rgb)

    detection = runtime.detector.detect(rgb_image, [target_name])
    if not detection.success or detection.num_objects == 0:
        return failed_place("Target detection failed")

    detection.set_random_obj_point(margin_ratio=0.5)
    target_pixel = detection.get_object_center()

    placing_result = plan_placing_point(
        rgb=rgb_image,
        depth_mm=frame.depth_mm,
        initial_pixel=target_pixel,
        target_name=target_name,
        intrinsics=frame.intrinsics,
    )

    p_camera = placing_result.p_camera
    p_world_h = runtime.T_world_camera @ p_camera
    p_world_place = p_world_h[:3].copy()
    p_world_place[0] += runtime.place_offset_world_x_m

    runtime.save_place_result(
        frame=frame,
        detection=detection,
        placing_result=placing_result,
        p_world_place=p_world_place,
    )
    runtime.set_place_target(target_name, detection, p_world_place)

    action = Action(
        id=runtime.next_action_id(),
        name="place",
        args={"object": object_name, "target": target_name},
    )
    return runtime.executor.execute_action(action)
```

pipeline 层应保持接近以上结构。HTTP request 校验、机器人 command、stop 和
具体运动步骤留在 client/backend 内部，不扩散到 `execute_place()`。

## `execute_pick` 标准流程

```text
获取 SceneSnapshot
  -> FineTunedYoloDetector 检测 object_name
  -> 选择 object bbox
  -> GraspNet 使用 bbox workspace 推理
  -> 过滤并选择 grasp candidate
  -> 写入 context.bboxes 和 context.grasps
  -> ActionExecutor.execute_action(pick)
  -> 关闭夹爪
  -> runtime 记录 held object 和 pick state
```

成功意味着：

- X5 到达 grasp pose；
- gripper close command 成功；
- runtime 和 backend 保留后续 place 所需状态。

## `execute_pipeline` 标准流程

```text
初始化 runtime 和 backend 一次
  -> execute_pick(runtime, object_name)
  -> pick 失败则立即停止
  -> execute_place(runtime, object_name, target_name)
  -> 写入 ExecutionReport
  -> shutdown runtime
```

公开 API：

```python
with create_x5_execution_runtime(config) as runtime:
    pick_result = execute_pick(runtime, object_name="number_block_3")
    place_result = execute_place(
        runtime,
        object_name="number_block_3",
        target_name="yellow_box",
    )
```

便捷 API：

```python
report = execute_pipeline(
    object_name="number_block_3",
    target_name="yellow_box",
    config_path="configs/execution/x5_pipeline.yaml",
    execute=False,
)
```

真机命令必须显式包含 `--execute`：

```bash
python -m agenticlab_human.execution.pipeline pipeline \
  --object number_block_3 \
  --target yellow_box \
  --config configs/execution/x5_pipeline.yaml \
  --execute
```

第一阶段不提供独立 place 进程。新进程无法可靠恢复 pick retreat pose 和
held-object 状态。需要独立 place 时，必须先设计可持久化且经过校验的
pick-state contract。

## 计划文件

```text
src/agenticlab_human/
├── execution/
│   ├── pipeline.py
│   ├── pipeline_types.py
│   └── place_target.py
└── perception/
    ├── detection/
    │   └── fine_tuned_yolo_detector.py
    └── grasping/
        └── http_backend.py

configs/execution/
└── x5_pipeline.yaml
```

职责：

- `pipeline.py`：runtime 生命周期、`capture_scene_from_x5_server()`、
  pick/place 编排和 CLI；
- `pipeline_types.py`：`SceneSnapshot` 和 run manifest；
- `place_target.py`：depth patch、反投影和 point transform；
- `fine_tuned_yolo_detector.py`：固定类别 fine-tuned YOLO；
- `http_backend.py`：`GraspNetHTTPClient` 到 `GraspBackend` 的 adapter。

不要把 YOLO、GraspNet 或坐标计算塞进 `action.py`。

## 标准配置

```yaml
pipeline:
  output_dir: output/execution
  x5_server_url: http://192.168.1.15:8000
  grasp_server_url: http://127.0.0.1:8010
  robot_config: configs/robot/x5_config.yaml
  camera_config: configs/perception/camera_config.yaml
  camera_name: Gemini335
  place_depth_patch_px: 9
  place_offset_world_x_m: null
  request_timeout_s: 120.0

detector:
  type: fine_tuned_yolo
  model_path: data/checkpoints/<fine-tuned-model>.pt
  confidence: 0.25
  image_size: 960

x5_place:
  default_place_orientation_rotvec: [rx, ry, rz]
  place_approach_offset_x_m: -0.05
```

真机 `--execute` 前必须满足：

- fine-tuned YOLO checkpoint 存在；
- `place_offset_world_x_m` 是经过标定的有限值；
- `default_place_orientation_rotvec` 是三个有限值；
- `T_world_camera` 是合法的齐次变换；
- X5 和 GraspNet health check 成功。

`place_offset_world_x_m: null` 只允许用于不发送运动命令的 perception 和
geometry 检查。

## 运行产物

每次运行创建：

```text
output/execution/<timestamp>/
├── run.json
├── pick/
│   ├── <frame>_rgb.png
│   ├── <frame>_depth_mm.npy
│   ├── <frame>_metadata.json
│   ├── detection.json
│   └── grasp_response.json
├── place/
│   ├── <frame>_rgb.png
│   ├── <frame>_depth_mm.npy
│   ├── <frame>_metadata.json
│   ├── detection.json
│   └── target_pose.json
└── execution_report.json
```

manifest 至少记录：

- object 和 target；
- 配置路径和服务 URL；
- frame ID 和 timestamp；
- bbox、随机 place pixel 和有效 depth；
- selected grasp；
- `T_camera_grasp`、`T_world_camera` 和 `T_world_tcp`；
- `p_camera`、`p_world_target` 和 `p_world_place`；
- `place_offset_world_x_m`；
- 每个 robot HTTP request ID；
- 最终结果或失败阶段。

## 失败处理

以下情况必须在机器人运动前失败：

- X5 或 GraspNet health check 失败；
- RGB-D 不合法或未同步；
- detector checkpoint 不存在；
- 请求类别不在 fine-tuned YOLO classes 中；
- object 或 target 检测失败；
- 没有可用 grasp；
- target depth 无效；
- `T_world_camera` 或计算结果存在非有限值；
- place 没有同一 runtime 中成功的 pick；
- 真机执行没有标定的 world-X offset 或 place orientation。

机器人开始运动后：

- 返回 backend 的 `failed_step`；
- backend 按现有规则发送 X5 stop；
- 不自动重试物理运动；
- 保存失败前产生的全部产物；
- 再次真机执行前需要 operator 确认。

## 最小实现顺序

1. 用 `SceneSnapshot` 包装现有 capture 和保存结果。
2. 实现 `execute_pick()` 和 `execute_place()`，跑通闭环。
3. 将 detector 的公开命名收敛为 `FineTunedYoloDetector`。
4. 实现 GraspNet adapter 和 place target 几何函数。
5. 增加 X5 world-X offset、固定 orientation 和 dry-run CLI。
6. 完成 mock HTTP 测试并分阶段验证真机：
   - capture；
   - detection 和 place point geometry；
   - grasp request；
   - pick、place 和 gripper。

`RemoteX5SceneProvider` 延后到多数据源或 capture 逻辑明显复杂时再实现。

## 验收标准

一条命令能够：

1. 接收显式 object 和 target 类别；
2. 不调用 TaskParser、PDDL、LLM、VLM 或 ActionChecker；
3. 获取 pick 和 place 两次对齐的 RGB-D；
4. 使用指定 fine-tuned YOLO checkpoint 检测 object 和 target；
5. 从 GraspNet 获得 `T_camera_grasp`；
6. 从target pixel 和 depth 计算有限的 `p_camera`；
7. 只通过 `T_world_camera` 计算 `p_world_target`；
8. 沿 world-X 应用已标定 place offset；
9. 使用固定 X5 place orientation；
10. 执行 `pick -> close -> retreat -> place -> open -> release retreat`；
11. 第一处失败时停止；
12. 保存足够产物复现 perception、frame transform 和 motion 决策。

## 后续工作

- container、surface 和 stacking 的特殊 place policy；
- VLM 语义验证；
- 自动动作效果检查；
- 物理运动自动重试；
- full-scene 多物体 grasp cache；
- 跨进程持久化 pick state；
- 多机器人通用配置。
