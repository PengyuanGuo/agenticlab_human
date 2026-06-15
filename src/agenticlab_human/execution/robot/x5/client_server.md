# AgenticLab X5 Client/Server

## 目标

Inference Client 执行 planner 生成的 `ActionSequence`，负责感知、grasp inference
和语义 action 到低层 robot command 的转换。公控机 Server 只负责：

1. 按请求采集一帧对齐的 RGB-D。
2. 接收经过验证的低层机器人指令。
3. 调用相机、X5 和夹爪硬件。
4. 返回真实执行结果和机器人状态。

当前不做实时视频流，也不把 planner、YOLO 或 AnyGrasp 放到公控机。

## 架构

```text
Inference Client                         Server PC

Planner -> ActionSequence
              |
ExecutionContext                         FastAPI
  | POST /v1/camera/capture ------------> Mock/Orbbec camera
  | <--------- RGB-D NPZ ----------------|
  |
YOLO -> grasp inference
  |
RemoteX5ActionBackend
  | POST /v1/robot/command -------------> Mock/X5 controller
  | <----- command result + state --------|
```

相机、机械臂和夹爪各自使用一个单线程 executor。`pyorbbecsdk`、`xapi`
和串口夹爪的初始化、调用和 shutdown 都保持在各自固定的线程中。

## 实现状态

各文件的稳定职责见
[Code Inventory](../../../../../docs/code_inventory.md#x5-clientserver)。

当前已实现：

- FastAPI server、同步 HTTP client、RGB-D NPZ 协议和 robot command contracts。
- 真实 Orbbec、真实 X5、串口夹爪，以及测试用 mock controller。
- `get_state`、`move_joints`、`movej_point`、`movel_point`、`stop` 和
`set_gripper`。
- tool-frame 初始化与读回校验，以及速度、关节范围和单次关节变化量校验。
- Remote pick/place：
`home -> approach -> grasp -> close -> retreat -> check gripper -> preplace -> place -> open -> home`。

相机验证：

- 2026-06-06 localhost 连续 capture `20/20` 成功，延迟中位数
`112.59 ms`，RGB/depth 同步差值中位数 `2.192 ms`。
- 2026-06-08 双机 capture 成功：Client 从 `192.168.1.15:8000` 收到
`uint8[720,1280,3]` RGB 和 `float32[720,1280]` 毫米深度，并保存完整
capture artifacts；该帧 RGB/depth 同步差值为 `14.312 ms`。

## HTTP API

### `GET /v1/health`

返回 camera 和 robot backend 是否 ready。

```json
{
  "status": "ok",
  "api_version": "v1",
  "camera": {"ready": true, "backend": "mock", "detail": "ready"},
  "robot": {"ready": true, "backend": "mock", "detail": "ready"}
}
```

### `POST /v1/camera/capture`

请求不需要 body。响应类型为 `application/x-npz`，包含：


| Field          | Type           | Meaning                         |
| -------------- | -------------- | ------------------------------- |
| `rgb`          | `uint8[H,W,3]` | RGB image                       |
| `depth_mm`     | `float32[H,W]` | aligned depth in millimeters    |
| `intrinsics`   | `float64[6]`   | `fx, fy, cx, cy, width, height` |
| `timestamp_ns` | `int64`        | server capture timestamp        |
| `frame_id`     | string         | unique frame identifier         |
| `wire_version` | `uint8`        | payload format version          |


NPZ 避免将 RGB/depth 展开成体积很大的 JSON list。

### `POST /v1/robot/command`

读取状态：

```json
{
  "request_id": "read-state-001",
  "command": {
    "type": "get_state",
    "arm": "all"
  }
}
```

执行关节目标：

```json
{
  "request_id": "move-left-001",
  "command": {
    "type": "move_joints",
    "arm": "left",
    "joints_rad": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    "speed_ratio": 0.1,
    "wait": true
  }
}
```

关节轨迹移动到 world-frame TCP Point：

```json
{
  "request_id": "movej-point-left-001",
  "command": {
    "type": "movej_point",
    "arm": "left",
    "tcp_pose_xyz_rotvec": [0.30, -0.45, 0.30, 0.0, 0.0, 1.57],
    "speed_ratio": 0.05,
    "wait": true
  }
}
```

直线移动到 world-frame TCP Point：

```json
{
  "request_id": "movel-point-left-001",
  "command": {
    "type": "movel_point",
    "arm": "left",
    "tcp_pose_xyz_rotvec": [0.30, -0.45, 0.295, 0.0, 0.0, 1.57],
    "speed_ratio": 0.03,
    "wait": true
  }
}
```

控制唯一夹爪：

```json
{
  "request_id": "close-gripper-001",
  "command": {
    "type": "set_gripper",
    "position": 0.0,
    "wait": true
  }
}
```

`position=0.0` 为闭合，`position=1.0` 为完全打开。命令不带 `arm`，
因为当前系统只有一个夹爪。

停止：

```json
{
  "request_id": "stop-001",
  "command": {
    "type": "stop",
    "arm": "all"
  }
}
```

成功响应包含：

- 原样返回的 `request_id`。
- 服务端接受的 `accepted_command`。
- `state_before` 和 `state_after`。
- `success`、`duration_ms`、服务端时间戳和错误信息。

所有公共协议使用：

- joint：弧度。
- Cartesian command position：米。
- Cartesian command orientation：rotation vector radians。
- Cartesian state orientation：quaternion `xyzw`。
- depth：毫米。

Note: X5 所需的毫米、角度和 SDK 对象只能在真实 controller 内转换。

## 安装与运行

项目依赖：

```bash
python -m pip install -e ".[dev,x5]"
```

`pyorbbecsdk` 和 X5 `xapi` 使用各自 SDK 的安装方式，不由 PyPI extra 安装。
Server PC 必须在能够运行现有 `cam_capture.py` 和 `test_x5_server.py` 的环境中
启动 server。

启动真实 Orbbec + real X5 server：

```bash
conda activate <env-with-pyorbbecsdk-and-xapi>
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_config.yaml
```

默认监听 `0.0.0.0:8000`。可使用命令行覆盖：

```bash
python -m agenticlab_human.execution.robot.x5.server \
  --host 127.0.0.1 \
  --port 8000
```

运行 HTTP mock 测试：

```bash
pytest -q tests/test_x5_http.py
```

FastAPI 自动 API 页面：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

独立 Client PC 只需要 AgenticLab X5 HTTP client，不需要安装
`pyorbbecsdk` 或 X5 `xapi`。将 `127.0.0.1` 换成公控机局域网 IP：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://<server-ip>:8000 \
  --save-dir output/x5_http_captures/server_pc \
  --preview
```

本机客户端保存一帧：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://192.168.1.15:8000 \
  --save-dir output/x5_http_captures/local
```

保存并显示：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://192.168.1.15:8000 \
  --save-dir output/x5_http_captures/local \
  --preview
```

客户端 Python API：

```python
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    print(client.health())

    frame = client.capture_rgbd()
    print(frame.rgb.shape, frame.depth_mm.shape, frame.intrinsics)

    client.close_gripper()
    client.open_gripper()
```

运行测试：

```bash
python -m pytest -q tests/test_x5_http.py tests/test_x5_controller.py
```

测试只使用 mock/fake xapi，不连接 Orbbec、X5 或夹爪。

## 当前执行顺序

Phase 3 real X5 路径：

```text
X5HTTPClient.get_state("left")
  -> POST /v1/robot/command
  -> robot executor
  -> RealX5Controller.get_state()
  -> x5.get_cjoint(), x5.get_wpoint(), x5.get_system_state()
  -> degrees/mm -> radians/meters/quaternion
  -> RobotCommandResponse

X5HTTPClient.stop("left")
  -> POST /v1/robot/command
  -> robot executor
  -> x5.stop()
  -> x5.wait_cmd_send_done()
  -> x5.abort()
  -> x5.wait_cmd_send_done()
  -> x5.wait_move_done()
  -> RobotCommandResponse

X5HTTPClient.move_joints("left", joints_rad=current + small_delta)
  -> POST /v1/robot/command
  -> read current joints from x5
  -> server-side speed / joint limit / max delta checks
  -> radians -> degrees -> x5.Joint
  -> x5.movj(handle, target, x5.MovPointAdd(vel, acc))
  -> optional x5.wait_move_done()
  -> RobotCommandResponse
```

Cartesian Point 路径：

```text
X5HTTPClient.movej_point()/movel_point()
  -> world TCP [x,y,z,rx,ry,rz] in meters/radians
  -> server-side speed / translation / rotation checks
  -> get_wpoint() for current cfg and current world TCP
  -> get_cjoint() for current R1/e2/e3
  -> rotvec -> Euler XYZ degrees
  -> x5.Point(..., uf=0, tf=active TF, cfg=current cfg)
  -> x5.movj(Point) or x5.movl(Point)
  -> optional x5.wait_move_done()
```

单夹爪路径：

```text
X5HTTPClient.close_gripper()/open_gripper()
  -> set_gripper(position=0.0/1.0)
  -> POST /v1/robot/command
  -> server routes SetGripperCommand to GripperService
  -> GripperService maps 0.0~1.0 to closed_position/open_position
  -> GripperController.set_position(raw_position)
  -> poll get_grip_status() until motion completes or timeout
```

### Phase 2：双机相机验证

已完成。当前记录的可复现实测命令：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://192.168.1.15:8000 \
  --save-dir output/x5_http_captures/local \
  --preview
```

下一步不再继续扩展相机路径，除非后续 grasp inference 需要额外 metadata。

### Phase 3：真实 X5 基础控制

标准工作流：

1. 在 Server PC 上确认 X5 API 可以单独连接：

```bash
python src/agenticlab_human/execution/robot/x5/test_x5_server.py
```

预期至少能看到有效 handle、版本信息和一次 IO 读取结果。

1. 检查 `configs/robot/x5_config.yaml`：

- `robot.backend: "x5"`。
- `robot.left.robot_ip` 为 X5 控制器 IP，当前为 `192.168.1.7`。
- `robot.left.tool_frame.tf_no` 为 active tool frame，当前设为 `1`。
- `tool_frame.position_m: [0, 0, 0.16]` 表示工具原点沿法兰 Z 轴偏移
`160 mm`；`rpy_deg` 当前为零旋转。
- `home_joints_deg` 只作为已知安全参考位，不作为首次自动运动目标。
- `head_joints_deg` 为 arm-only `move_joints` 下发时保留的 head joint 1/2，
当前默认 `[0, 28]`。
- `max_command_speed_ratio` 当前为 `0.10`。
- `max_joint_delta_deg` 当前为 `5.0`，防止第一阶段误发大幅关节目标。

1. 在 Server PC 启动真实 Orbbec + real X5 server：

```bash
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_config.yaml \
  --host 0.0.0.0 \
  --port 8000
```

如果本轮只调 X5，而当前 X5 `xapi` 环境还没有 Orbbec SDK，可以临时把
`x5_config.yaml` 的 `camera.backend` 改成 `mock`，先完成 robot-only 验证。

1. 在 Client PC 先做 health 和 get_state，不发运动：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    print(client.health())
    result = client.get_state("left")
    print(result.success)
    state = result.state_after.arms["left"]
    print("joints_rad:", state.joints_rad)
    print("world_tcp_xyzw:", state.tcp_pose_xyzw)
    print("tool_frame_no:", state.tool_frame_no)
    print("tool_frame_pose_xyzw:", state.tool_frame_pose_xyzw)
PY
```

1. 单独验证 stop：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    result = client.stop("left")
    print(result.success, result.error)
PY
```

1. 已确认急停、限位和周围空间安全后，从当前 state 构造小幅 joint target：

```bash
python - <<'PY'
import math
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    state = client.get_state("left").state_after
    joints = list(state.arms["left"].joints_rad)
    joints[0] += math.radians(1.0)
    result = client.move_joints(
        "left",
        joints,
        speed_ratio=0.05,
        wait=True,
        request_id="phase3-safe-j1-plus-1deg",
    )
    print(result.success, result.error)
    print(result.state_after.arms["left"].joints_rad)
PY
```

注意：第一次运动不要发送 `[0.0] * 7`，也不要直接发送 home joints。先读当前
joint，再只对一个关节增加 `1 deg` 左右。如果服务端返回
`max delta ... exceeds configured max_joint_delta_deg`，说明安全校验生效。

1. 真机 `movej(Point)` 当前位姿零移动。Client state 返回 quaternion，

command 使用 rotvec，因此在 Client PC 转换：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    tcp_pose_xyzw_to_xyz_rotvec,
)

with X5HTTPClient("http://192.168.1.15:8000") as client:
    state = client.get_state("left").state_after.arms["left"]
    pose6d = tcp_pose_xyzw_to_xyz_rotvec(state.tcp_pose_xyzw)
    result = client.movej_point(
        "left",
        pose6d,
        speed_ratio=0.03,
        wait=True,
        request_id="phase3-movej-point-zero",
    )
    print(result.success, result.error)
    print(result.state_after.arms["left"].tcp_pose_xyzw)
PY
```

预期机械臂、R1 和 head joints 均不发生可见移动。这一步用于确认 `Point` 的
`uf=0/tf=1/cfg/e1/e2/e3` 构造正确。

1. 零移动通过后，只沿已确认安全的 world axis 增加 `5 mm`，执行

`movej(Point)`：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    tcp_pose_xyzw_to_xyz_rotvec,
)

with X5HTTPClient("http://192.168.1.15:8000") as client:
    state = client.get_state("left").state_after.arms["left"]
    pose6d = tcp_pose_xyzw_to_xyz_rotvec(state.tcp_pose_xyzw)
    pose6d[2] += 0.005
    result = client.movej_point(
        "left", pose6d, speed_ratio=0.03, wait=True,
        request_id="phase3-movej-point-z-plus-5mm",
    )
    print(result.success, result.error)
PY
```

1. 回到已确认起点后，以同样的 `5 mm` 目标测试 `movel(Point)`：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    tcp_pose_xyzw_to_xyz_rotvec,
)

with X5HTTPClient("http://192.168.1.15:8000") as client:
    state = client.get_state("left").state_after.arms["left"]
    pose6d = tcp_pose_xyzw_to_xyz_rotvec(state.tcp_pose_xyzw)
    pose6d[2] += 0.005
    result = client.movel_point(
        "left", pose6d, speed_ratio=0.02, wait=True,
        request_id="phase3-movel-point-z-plus-5mm",
    )
    print(result.success, result.error)
PY
```

注意：示例使用 world Z 只是为了展示格式。真机测试必须根据现场机械臂姿态、
桌面和障碍物选择明确安全的方向，并保持急停可用。

AnyGrasp 到 X5 TCP 的姿态规划约定：

```text
T_grasp_tcp = T_grasp_ee

R_grasp_tcp =
[[ 0, 0, 1],
 [ 0, 1, 0],
 [-1, 0, 0]]

T_world_tcp = T_world_camera @ T_camera_grasp @ T_grasp_tcp
```

approach pose 沿 AnyGrasp grasp frame 的 `-X` 方向退开
`approach_distance_m`，然后使用：

```text
movej_point(approach)
movel_point(grasp)
close_gripper()
movel_point(approach)
```

### Phase 3.5：Remote pick/place trajectory

`x5_config.yaml` 的 Client 侧配置：

```yaml
action_backend:
  server_url: "http://192.168.1.15:8000"
  arm: "left"
  camera_name: "Gemini335"
  approach_distance_m: 0.05
  home_speed_ratio: 0.05
  home_max_step_deg: 4.0
  approach_speed_ratio: 0.03
  grasp_speed_ratio: 0.02
  retreat_speed_ratio: 0.02
  place_approach_speed_ratio: 0.03
  place_speed_ratio: 0.02
  place_approach_offset_x_m: -0.05
  default_place_orientation_rotvec: [1.2172784, 1.2123690, 1.2159012]
  request_timeout_s: 90.0
```

`RemoteX5ActionBackend` 只供生产 pipeline 真机执行，不再提供独立 CLI、
dry-run 或 staged validation。运行入口是：

```bash
python -m agenticlab_human.execution.pipeline pipeline \
  --object number_block_3 \
  --target yellow_bin \
  --config configs/execution/x5_pipeline.yaml \
  --execute
```

完整物理动作：

```text
home -> approach -> grasp -> close gripper
retreat -> check gripper -> preplace -> place -> open gripper -> home
```

`place_approach_offset_x_m` 是带符号的 world-X offset：

```text
pre_place.x = place.x + place_approach_offset_x_m
pre_place.y = place.y
pre_place.z = place.z
```

preplace/place 使用 `default_place_orientation_rotvec`：

```text
place_pose = [target_x, target_y, target_z, rx, ry, rz]
pre_place_pose = [
  target_x + x_offset,
  target_y,
  target_z,
  rx,
  ry,
  rz,
]
```

`target_pose` 必须是 world-frame pose vector（至少包含 XYZ）或 `4x4`
world transform。target 自带的 orientation 会被忽略。

### Phase 4：AgenticLab 接入

1. `scene_provider.py` 实现 `RemoteRGBDSceneProvider.capture_rgbd()`。
2. 将 remote scene provider 交给 `ExecutionContext`。
3. Client 本地运行 YOLO 和 grasp backend。
4. 将 `RemoteX5ActionBackend` 接入正式 action execution 配置。
5. 真机验证 gripper HTTP command、pick close 和 place open。

### Phase 5：安全与可靠性

1. request ID 去重，防止超时重试导致重复运动。
2. joint/workspace/speed limits 服务端强制校验。
3. command timeout、busy state 和明确的 stop 行为。
4. 日志记录 request、command、state 和执行耗时。
5. 部署网络认证；当前 API 不应直接暴露到非可信网络。

只有出现连续 `servoj/servol` 高频控制需求时才增加 WebSocket。当前离散
action 必须等待服务端返回结果，HTTP request-response 更合适。

## Reference

- FastAPI tutorial: [https://fastapi.tiangolo.com/tutorial/](https://fastapi.tiangolo.com/tutorial/)
- Lifespan events: [https://fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/)
- Testing: [https://fastapi.tiangolo.com/tutorial/testing/](https://fastapi.tiangolo.com/tutorial/testing/)
- Direct responses: [https://fastapi.tiangolo.com/advanced/response-directly/](https://fastapi.tiangolo.com/advanced/response-directly/)

