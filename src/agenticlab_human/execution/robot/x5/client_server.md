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

相机和机器人各自使用一个单线程 executor。以后接入 `pyorbbecsdk` 和
`xapi` 时，初始化、调用和 shutdown 都保持在各自固定的线程中。

## 实现状态

Phase 1 已完成：

- `contracts.py`
  - FastAPI/Pydantic request-response models。
  - RGB-D NPZ 编解码。
  - `get_state`、`move_joints`、`stop` command contracts。
- `camera.py`
  - `RGBDCamera` protocol。
  - deterministic `MockRGBDCamera`。
- `x5_controller.py`
  - `X5Controller` protocol。
  - 双臂 `MockX5Controller`。
- `server.py`
  - FastAPI lifespan 初始化与清理。
  - health、capture 和 robot command endpoints。
  - mock config 启动入口。
- `client.py`
  - 同步 `X5HTTPClient`。
  - health、RGB-D capture、get state、move joints 和 stop methods。
- `tests/test_x5_http.py`
  - mock health、图像往返、robot command 和请求校验测试。

Phase 2 本机验证已完成：

- `OrbbecRGBDCamera` 已实现，并适配现有 `CameraCapture`。
- `CameraCapture.capture_with_metadata()` 返回：
  - 对齐的 RGB `uint8[H,W,3]`。
  - 深度 `float32[H,W]`，单位毫米。
  - 当前实际 stream profile 的 runtime intrinsics。
  - color/depth system timestamp 和同步差值。
- Phase 2 验证时使用真实 Orbbec camera 与 mock robot。
- `x5_config.yaml` 现为真实 Orbbec camera + real X5 controller。
- `x5_mock_config.yaml` 保留纯 mock 测试配置。
- HTTP client 可以保存 RGB、原始 float32 depth、16-bit depth PNG、
  depth preview 和 metadata JSON。
- HTTP client 支持 `--preview` 显示收到的 RGB/depth。

2026-06-06 本机实测：

- RGB 和 depth 均为 `1280x720`。
- Runtime intrinsics：
  `fx=612.6924, fy=612.5692, cx=640.1061, cy=360.9931`。
- Depth dtype 为 `float32`，单位为 `mm`。
- 一次 HTTP capture 的 RGB/depth timestamp 差值为 `2.21 ms`。
- 保存目录：
  `output/x5_http_captures/real_orbbec_local_final/`。
- 连续 20 次 localhost HTTP capture：
  - 成功率 `20/20`，frame ID 全部唯一。
  - 延迟 `min=110.22 ms, median=112.59 ms, mean=113.24 ms, max=124.16 ms`。
  - RGB/depth 同步差值 `min=2.087 ms, median=2.192 ms, max=2.333 ms`。

2026-06-08 双机 capture 实测已完成：

- Client PC 通过 `http://192.168.1.15:8000` 请求 Server PC。
- Server health 返回 `camera=orbbec robot=mock`。
- 收到 RGB `uint8[720,1280,3]` 和 depth `float32[720,1280]`。
- Runtime intrinsics：
  `fx=612.9034, fy=612.8455, cx=649.5079, cy=367.4934`。
- RGB/depth timestamp 差值为 `14.312 ms`。
- Client PC 已保存 RGB、depth PNG、depth NPY、depth preview 和 metadata。

Phase 3 基础控制代码已实现：

- `RealX5Controller(X5Controller)`。
- `server.py` 支持 `robot.backend: "x5"`。
- `x5_config.yaml` 加载 robot IP、home joints、速度限制、关节限制和单次最大关节变化量。
- `RealX5Controller` lazy import `xapi.api`，因此 Client PC 不需要安装 X5 SDK。
- `get_state` 读取 `x5.get_cjoint()` 和 `x5.get_cpoint()`，对外返回弧度、米和 quaternion `xyzw`。
- `stop` 调用 `stop -> wait_cmd_send_done -> abort -> wait_cmd_send_done -> wait_move_done`。
- `move_joints` 使用 `x5.Joint` 和 `x5.MovPointAdd(vel, acc)`，并在服务端强制校验速度、joint limits 和最大 delta。
- Client 端 `move_joints` 仍只发送 7 个 arm joints；Server 端下发 xapi 时从
  `head_joints_deg` 补齐 head joint 1/2，避免 arm-only move 把 head 轴归零。

暂未实现或验证：

- `scene_provider.py` 的 Remote RGB-D `SceneProvider`。
- `remote_backend.py` 的 AgenticLab `ActionBackend`。
- Phase 3 真实机械臂上的 `get_state`、`stop` 和小幅 `move_joints` 实测。
- 夹爪 driver。
- TCP/gripper/home commands。

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
- Cartesian position：米，后续加入。
- orientation：quaternion `xyzw`，后续加入。
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

启动纯 mock server：

```bash
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_mock_config.yaml
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
```

运行测试：

```bash
python -m pytest -q tests/test_x5_http.py tests/test_x5_controller.py
```

测试只使用 mock/fake xapi，不连接 Orbbec、X5 或夹爪。

## 当前执行顺序

Phase 1 mock 路径：

```text
X5HTTPClient.capture_rgbd()
  -> POST /v1/camera/capture
  -> camera executor
  -> MockRGBDCamera.capture()
  -> RGBDFrame -> NPZ
  -> client decode -> RGBDFrame

X5HTTPClient.move_joints()
  -> Pydantic RobotCommandRequest
  -> POST /v1/robot/command
  -> read state_before
  -> robot executor
  -> MockX5Controller.execute()
  -> read state_after
  -> RobotCommandResponse
```

Phase 3 real X5 路径：

```text
X5HTTPClient.get_state("left")
  -> POST /v1/robot/command
  -> robot executor
  -> RealX5Controller.get_state()
  -> x5.get_cjoint(), x5.get_cpoint(), x5.get_system_state()
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

## 后续计划

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

2. 检查 `configs/robot/x5_config.yaml`：

- `robot.backend: "x5"`。
- `robot.left.robot_ip` 为 X5 控制器 IP，当前为 `192.168.1.7`。
- `home_joints_deg` 只作为已知安全参考位，不作为首次自动运动目标。
- `head_joints_deg` 为 arm-only `move_joints` 下发时保留的 head joint 1/2，
  当前默认 `[0, 28]`。
- `max_command_speed_ratio` 当前为 `0.10`。
- `max_joint_delta_deg` 当前为 `5.0`，防止第一阶段误发大幅关节目标。

3. 在 Server PC 启动真实 Orbbec + real X5 server：

```bash
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_config.yaml \
  --host 0.0.0.0 \
  --port 8000
```

如果本轮只调 X5，而当前 X5 `xapi` 环境还没有 Orbbec SDK，可以临时把
`x5_config.yaml` 的 `camera.backend` 改成 `mock`，先完成 robot-only 验证。

4. 在 Client PC 先做 health 和 get_state，不发运动：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    print(client.health())
    result = client.get_state("left")
    print(result.success)
    print(result.state_after.arms["left"].joints_rad)
PY
```

5. 单独验证 stop：

```bash
python - <<'PY'
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://192.168.1.15:8000") as client:
    result = client.stop("left")
    print(result.success, result.error)
PY
```

6. 已确认急停、限位和周围空间安全后，从当前 state 构造小幅 joint target：

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

下一步：

1. 完成真实机械臂 `get_state` 实测。
2. 完成真实机械臂 `stop` 实测。
3. 完成低速 `move_joints` 小幅实测。
4. 接入夹爪后增加 `set_gripper`。
5. 确认状态读取和运动指令全部在 robot executor 线程执行。

### Phase 4：AgenticLab 接入

1. `scene_provider.py` 实现 `RemoteRGBDSceneProvider.capture_rgbd()`。
2. 将 remote scene provider 交给 `ExecutionContext`。
3. Client 本地运行 YOLO 和 grasp backend。
4. `remote_backend.py` 实现 AgenticLab `ActionBackend`。
5. 将 grasp pose 转换为 approach、grasp、close、retreat command sequence。

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
