# 无规划执行流水线

现有的小拓相机（Gemini 335）、YOLO checkpoint、GraspNet HTTP 服务和 X5 HTTP
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

```yaml
detector:
  type: yolo
  # Use the trained closed-set tabletop checkpoint for real X5 execution.
  model_path: data/checkpoints/<fine-tuned-model>.pt
  confidence: 0.25
  image_size: 960
```

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

- `place_offset_world_x_m`：修正最终 place target；
- `place_approach_offset_x_m`：由最终 place pose 生成 preplace pose。

## 标准架构

```text
本地执行进程
├── ExecutionRuntime
│   ├── X5HTTPClient + capture_scene_from_x5_server()
│   ├── YOLODETECTOR
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

## `execute_place` 标准实现

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

## 计划文件

```text
src/agenticlab_human/
├── execution/
│   ├── pipeline.py
│   ├── pipeline_types.py
│   └── place_target.py
└── perception/
    ├── detection/
    │   └── yolo_detector.py
    └── grasping/
        └── http_backend.py

configs/execution/
└── x5_pipeline.yaml
```

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
  type: yolo
  # Use the trained closed-set tabletop checkpoint for real X5 execution.
  model_path: data/checkpoints/<fine-tuned-model>.pt
  confidence: 0.25
  image_size: 960

x5_place:
  default_place_orientation_rotvec: [rx, ry, rz]
  place_approach_offset_x_m: -0.05
```

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
- 请求类别不在 YOLO checkpoint classes 中；
- object 或 target 检测失败；
- 没有可用 grasp；
- target depth 无效；
- `T_world_camera` 或计算结果存在非有限值；
- place 没有同一 runtime 中成功的 pick；
- 真机执行没有标定的 world-X offset 或 place orientation。

机器人开始运动后：

- backend 按现有规则发送 X5 stop；
- 不自动重试物理运动；
- 保存失败前产生的全部产物；
- 再次真机执行前需要 operator 确认。

## 最小实现顺序

1. 用 `SceneSnapshot` 包装现有 capture 和保存结果。
2. 实现 `execute_pick()` 和 `execute_place()`，跑通闭环。
3. 实现 GraspNet adapter 和 place target 几何函数。
4. 增加 X5 world-X offset 和固定 orientation。
5. 完成 mock HTTP 测试并验证真机：
  - capture；
  - detection 和 place point geometry；
  - grasp request；
  - pick、place 和 gripper。

`RemoteX5SceneProvider` 延后到多数据源或 capture 逻辑明显复杂时再实现。

## 后续工作

- container、surface 和 stacking 的特殊 place policy；
- reset
- full-scene 多物体 grasp cache；

