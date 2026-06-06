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
- `x5_config.yaml` 当前使用真实 Orbbec camera 与 mock robot。
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

暂未实现或验证：

- `scene_provider.py` 的 Remote RGB-D `SceneProvider`。
- `remote_backend.py` 的 AgenticLab `ActionBackend`。
- 真实 X5/xapi controller 和夹爪 driver。
- TCP/gripper/home commands。
- Server PC 和独立 Client PC 之间的双机网络实测。

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

| Field | Type | Meaning |
| --- | --- | --- |
| `rgb` | `uint8[H,W,3]` | RGB image |
| `depth_mm` | `float32[H,W]` | aligned depth in millimeters |
| `intrinsics` | `float64[6]` | `fx, fy, cx, cy, width, height` |
| `timestamp_ns` | `int64` | server capture timestamp |
| `frame_id` | string | unique frame identifier |
| `wire_version` | `uint8` | payload format version |

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

X5 所需的毫米、角度和 SDK 对象只能在真实 controller 内转换。

## 安装与运行

项目依赖：

```bash
python -m pip install -e ".[dev,x5]"
```

`pyorbbecsdk` 使用相机 SDK 自己的安装方式，不由 PyPI extra 安装。公控机
必须在能够运行现有 `cam_capture.py` 的环境中启动 server。

启动真实 Orbbec + mock robot server：

```bash
conda activate xiaotuo_audio
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
`pyorbbecsdk`。将 `127.0.0.1` 换成公控机局域网 IP：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://<server-ip>:8000 \
  --save-dir output/x5_http_captures/server_pc \
  --preview
```

本机客户端保存一帧：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://127.0.0.1:8000 \
  --save-dir output/x5_http_captures/local
```

保存并显示：

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://127.0.0.1:8000 \
  --save-dir output/x5_http_captures/local \
  --preview
```

客户端 Python API：

```python
from agenticlab_human.execution.robot.x5.client import X5HTTPClient

with X5HTTPClient("http://127.0.0.1:8000") as client:
    print(client.health())

    frame = client.capture_rgbd()
    print(frame.rgb.shape, frame.depth_mm.shape, frame.intrinsics)

    result = client.move_joints(
        arm="left",
        joints_rad=[0.0] * 7,
        speed_ratio=0.1,
    )
    print(result)
```

运行测试：

```bash
python -m pytest -q tests/test_x5_http.py
```

测试只使用 mock，不连接 Orbbec、X5 或夹爪。

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

## 后续计划

### Phase 2：双机相机验证

本机真实相机 HTTP 路径已完成，剩余双机验证：

1. Server PC 使用 `x5_config.yaml` 启动并监听局域网地址。
2. 防火墙只开放配置的 API port 给 Client PC。
3. Client PC 使用 `http://<server-ip>:8000` 请求并保存一帧。
4. 比较 Server 本地 capture 和 Client 接收结果的 shape、dtype、intrinsics。
5. 记录传输耗时和连续 20 次 capture 的成功率。

### Phase 3：真实 X5 基础控制

1. 实现 `RealX5Controller(X5Controller)`。
2. 从 `x5_config.yaml` 加载 robot IP、home joints 和安全限制。
3. 首先实现 `get_state` 和 `stop`。
4. 再以低速执行已知安全的 `move_joints`。
5. 接入夹爪后增加 `set_gripper`。
6. 确认状态读取和运动指令全部在 robot executor 线程执行。

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

- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- Lifespan events: https://fastapi.tiangolo.com/advanced/events/
- Testing: https://fastapi.tiangolo.com/tutorial/testing/
- Direct responses: https://fastapi.tiangolo.com/advanced/response-directly/
