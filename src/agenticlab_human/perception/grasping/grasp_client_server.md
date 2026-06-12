# GraspNet Client/Server

## 目标

GraspNet 不进入 `agenticlab_human` 的主推理环境。它运行在已有的
`flexiv` conda 环境中，通过 localhost FastAPI 服务接收本机已经保存的
RGB-D 文件，返回 camera-frame grasp candidates。

```text
agenticlab_human env
  -> X5 HTTP capture 保存 RGB/depth
  -> YOLO bbox
  -> GraspNetHTTPClient
  -> POST /v1/grasp/predict

flexiv env
  -> FastAPI server
  -> startup 加载 GraspNet checkpoint 和 Gemini335 camera config
  -> RGB-D + bbox/mask
  -> GraspNet inference + collision filter + NMS
  -> camera-frame pose_4x4 candidates
  -> 可选：response 后启动独立 Open3D visualization subprocess

execution layer
  -> camera frame 转 robot base
  -> approach / grasp / retreat
```

服务端不处理 hand-eye transform，也不发送机器人指令。

## 可选 Open3D 可视化

默认 server 不打开窗口。调试时可增加：

```bash
OMP_NUM_THREADS=12 PYTHONPATH=src python -m \
  agenticlab_human.perception.grasping.server \
  --config configs/perception/graspnet_config.yaml \
  --camera-config configs/perception/camera_config.yaml \
  --camera-name Gemini335 \
  --checkpoint data/checkpoints/minkuresunet_realsense.tar \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8010 \
  --visualize-seconds 10
```

成功的 predict request 会按以下顺序运行：

```text
GraspNet inference
  -> FastAPI 返回 camera-frame grasps
  -> BackgroundTasks 启动独立 Python subprocess
  -> subprocess 重新读取 RGB-D 和相同 workspace mask
  -> Open3D 显示返回的 grasps
  -> 保存截图到 output/grasp_viz/
  -> 10 秒后 destroy_window() 并退出
```

可视化沿用旧 wrapper 的简单模式：workspace 彩色点云、
`GraspGroup.to_open3d_geometry_list()` grippers、原有相机旋转和截图裁白边。
它不在 CUDA inference executor 中运行，因此窗口等待不会占住 GraspNet
推理线程。没有 grasp candidate 时不会启动窗口。

## 当前实现

```text
contracts.py
  Pydantic health、predict request/response contract

backend.py
  GraspInferenceBackend protocol
  GraspNetInferenceBackend
  MockGraspInferenceBackend

graspnet_wrapper.py
  checkpoint 加载
  从 camera_config.yaml 读取 Gemini335 intrinsics
  RGB-D + workspace mask 转点云
  GraspNet inference
  collision filtering / NMS / score sorting
  X5 approach-axis angle filtering
  camera-frame 4x4 candidates

server.py
  GET  /v1/health
  POST /v1/grasp/predict
  单线程 executor 保持模型初始化和 CUDA inference 在同一线程

client.py
  GraspNetHTTPClient
  本机 capture 文件路径自动推导
  response 转 GraspCandidate

grasp_backend.py
  execution layer 使用的 GraspCandidate / GraspBackend contract
```

旧路径 `agenticlab_human.perception.backend.grasp_backend` 保留了一个兼容
转发文件，现有 execution 代码不需要立即批量修改 import。

## 文件输入

X5 camera client 已经把同一帧保存成：

```text
orbbec-0000641176_rgb.png
orbbec-0000641176_depth_mm.npy
orbbec-0000641176_depth_mm.png
```

Grasp client 只需要传 `_rgb.png`。它会根据文件名自动推导：

```text
*_depth_mm.npy
```

GraspNet server 不从 capture metadata 接收动态 intrinsics。启动时固定读取：

```yaml
# configs/perception/camera_config.yaml
Gemini335:
  intrinsic_matrix:
  - [612.903442, 0.0, 649.507935]
  - [0.0, 612.845459, 367.493439]
  - [0.0, 0.0, 1.0]
  factor_depth: 1000.0
  width: 1280
  height: 720
```

输入 RGB/depth 分辨率必须和该 profile 的 `1280x720` 一致。

Depth contract 固定为：

```text
float32[H,W]
unit: millimeter
```

## HTTP API

### `GET /v1/health`

```json
{
  "status": "ok",
  "api_version": "v1",
  "backend": "graspnet",
  "model_loaded": true,
  "device": "cuda:0",
  "detail": "ready"
}
```

### `POST /v1/grasp/predict`

最小请求：

```json
{
  "rgb_path": "/abs/path/orbbec-0000641176_rgb.png",
  "depth_path": "/abs/path/orbbec-0000641176_depth_mm.npy",
  "bbox_xyxy": [558, 550, 608, 612],
  "object_label": "number-block-3",
  "depth_unit": "mm",
  "mask_offset_px": 10,
  "max_grasps": 10,
  "score_threshold": 0.0,
  "collision_detection": true,
  "nms": true
}
```

也可以传 `workspace_mask_path`。其优先级高于 bbox 生成的矩形 mask。
如果 mask 和 bbox 都没有提供，则使用整张有效深度图。

返回：

```json
{
  "success": true,
  "pose_frame": "camera",
  "object_label": "number-block-3",
  "grasps": [
    {
      "pose_4x4": [
        [0.0, 1.0, 0.0, -0.10],
        [-0.18, -0.09, 0.98, 0.33],
        [0.98, 0.0, 0.18, 0.93],
        [0.0, 0.0, 0.0, 1.0]
      ],
      "score": 0.51,
      "width": 0.078,
      "height": 0.02,
      "depth": 0.04,
      "object_label": "number-block-3",
      "image_xy": [583.0, 584.0],
      "metadata": {
        "source": "graspnet"
      }
    }
  ],
  "num_grasps": 1,
  "duration_ms": 408.0
}
```

`pose_4x4` 是 `T_camera_grasp`。`image_xy` 是 grasp translation 投影到 RGB
图像后的像素位置，可用于检查候选是否仍落在目标 bbox 内。

## 安装

`flexiv` 环境原本已有：

```text
PyTorch 2.4 CUDA 12.1
MinkowskiEngine
Open3D
graspnetAPI
OpenCV
```

HTTP server 还需要：

```bash
conda activate flexiv
python -m pip install fastapi uvicorn
```

项目 package 可以 editable install，也可以在 repo root 使用 `PYTHONPATH=src`。
最小验证使用后者，避免向 flexiv 环境加入主项目的全部依赖。

## 启动 Server

在 `flexiv` 环境中：

```bash
cd /home/agenticlab/Project/agenticlab_human
conda activate flexiv

OMP_NUM_THREADS=12 PYTHONPATH=src \
python -m agenticlab_human.perception.grasping.server \
  --config configs/perception/graspnet_config.yaml \
  --camera-config configs/perception/camera_config.yaml \
  --camera-name Gemini335 \
  --checkpoint data/checkpoints/minkuresunet_realsense.tar \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8010
```

检查：

```bash
curl http://127.0.0.1:8010/v1/health
```

FastAPI 页面：

```text
http://127.0.0.1:8010/docs
http://127.0.0.1:8010/redoc
```

## 调用 Client

在 `agenticlab_human` 主环境或当前开发环境中：

```bash
cd /home/agenticlab/Project/agenticlab_human

PYTHONPATH=src \
python -m agenticlab_human.perception.grasping.client \
  --server-url http://127.0.0.1:8010 \
  --rgb-path output/x5_http_captures/local/orbbec-0000641176_rgb.png \
  --bbox 558 550 608 612 \
  --object-label number-block-3 \
  --max-grasps 10
```

可选参数：

```text
--depth-path
--workspace-mask-path
--score-threshold
--mask-offset-px
--no-collision-detection
--no-nms
```

## Python 调用

```python
from agenticlab_human.perception.grasping.client import GraspNetHTTPClient


with GraspNetHTTPClient("http://127.0.0.1:8010") as client:
    result = client.predict_from_files(
        rgb_path="output/x5_http_captures/local/orbbec-0000641176_rgb.png",
        bbox_xyxy=[558, 550, 608, 612],
        object_label="number-block-3",
        max_grasps=10,
    )
    grasp_candidates = client.to_grasp_candidates(result)
```

`to_grasp_candidates()` 会写入：

```text
candidate.pose          = camera-frame 4x4 matrix
candidate.score         = GraspNet score
candidate.image_xy      = projected grasp center
candidate.object_name   = request object_label
candidate.metadata:
  frame: camera
  width
  height
  depth
  source: graspnet
```

这和 `RemoteX5ActionBackend` / `FlexivActionBackend` 当前期待的 camera-frame
`GraspCandidate` 对齐。

## 2026-06-10 最小真实验证

使用：

```text
RGB:      output/x5_http_captures/local/orbbec-0000641176_rgb.png
Depth:    output/x5_http_captures/local/orbbec-0000641176_depth_mm.npy
BBox:     [558, 550, 608, 612]
Object:   number-block-3
Camera:   Gemini335 from configs/perception/camera_config.yaml
```

结果：

```text
server health: ok
backend: graspnet
device: cuda:0
RGB/depth: 1280x720
valid depth points: 725161
workspace valid points: 5493
returned grasps: 10
top score: 0.513888
top width: 0.077983 m
top image_xy: [583, 584]
camera profile: Gemini335
request duration: 259.1 ms
```

top grasp 的投影点位于数字 3 方块 bbox 内。这个结果证明：

```text
本机已保存 RGB-D
  -> agenticlab_human client
  -> flexiv GraspNet server
  -> checkpoint/CUDA inference
  -> camera-frame grasp candidates
```

最小工作流已闭环，但还没有做 grasp 3D 可视化和机器人执行。

## 小 bbox 与 workspace offset

client 默认使用 `--mask-offset-px 10`，旧的
`graspnet_wrapper_with_custom_bbox.py` 默认使用 `--bbox_offset 20`。
因此只写相同 bbox，并不代表两条路径送入模型的是相同 workspace。

对 number-block-2：

```text
BBox: [481, 596, 521, 646]

offset 10:
  workspace valid points: 3960
  stage-1 graspable points: 0

offset 20:
  workspace valid points: 6960
  raw grasps: 1024
  grasps after collision filtering: 416
```

复现旧脚本行为时应显式传：

```bash
PYTHONPATH=src python -m agenticlab_human.perception.grasping.client \
  --server-url http://127.0.0.1:8010 \
  --rgb-path output/x5_http_captures/local/orbbec-0000641176_rgb.png \
  --bbox 481 596 521 646 \
  --mask-offset-px 20 \
  --object-label number-block-2 \
  --max-grasps 10
```

PointNet2 的 CUDA FPS 要求输入点数大于目标采样数。原模型在 stage-1
graspable 点少于固定的 `M_POINT=1024` 时仍直接调用 FPS，可能导致
`gather_points_kernel` illegal memory access。当前实现会对不足的点合法重复
补齐；零 graspable 点则返回空 candidates，不再破坏 server 的 CUDA context。

## X5 grasp approach angle filter

候选在 collision filtering、NMS、score threshold 和 score sorting 后，会执行
X5 专用方向筛选：

```text
R_base_grasp = R_base_camera @ R_camera_grasp
approach_base = R_base_grasp[:, 0]
angle = angle(approach_base, robot_base_+X)
```

默认范围来自：

```yaml
# configs/perception/graspnet_config.yaml
angle_threshold: 30
```

即保留 `0° <= angle <= 30°` 的 top-50 候选，并维持原 score 排序。如果
范围内没有候选，则返回夹角最接近范围中心的一个候选，避免把全部候选清空。

## 下一步

1. 把 YOLO 返回的真实 bbox 直接传给 `predict_from_files()`。
2. 保存或显示点云和 top-N grasp geometry，确认姿态方向。
3. 在 `ExecutionContext` 的 object-level refresh 中调用 GraspNet client。
4. 由 execution backend 完成 camera-to-base transform 和真实抓取。
