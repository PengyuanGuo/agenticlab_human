# AgenticLab YOLO Fine-tuning Guide

本文档记录 AgenticLab 桌面机器人目标检测数据集的完整路线：

```text
camera/video data collection
  -> preprocess
  -> pre-label
  -> human correction
  -> active learning
  -> pseudo-labeling
  -> YOLO fine-tuning
  -> evaluation
  -> ONNX/TensorRT export
  -> AgenticLab deployment
  -> retrain
```

目标是训练一个可部署到 `agenticlab_human` perception stack 的闭集 YOLO detector。部署后它会输出 bbox、label、score、center point，并由 `src/agenticlab_human/perception/detection/yolo_world_detector.py` 转成下游 grasp/planning 可以使用的 detection result。

本文以 `yolo26s.pt` 为例。官方参考：

- Ultralytics train mode: <https://docs.ultralytics.com/modes/train/>
- Ultralytics detect task: <https://docs.ultralytics.com/tasks/detect/>
- Ultralytics YOLO dataset format: <https://docs.ultralytics.com/datasets/detect/>
- Ultralytics export mode: <https://docs.ultralytics.com/modes/export/>
- Ultralytics Platform annotation: <https://docs.ultralytics.com/platform/data/annotation/>
- Ultralytics SAM auto annotation: <https://docs.ultralytics.com/models/sam/>
- Ultralytics CLI usage: <https://docs.ultralytics.com/usage/cli/>

## 0. 三个明确结论

### 0.1 同一个物体要不要多个 synonym label？

不建议。

同一个可操作物体必须只有一个 canonical class name。比如同一个数字积木不要同时标成：

```text
number block 3
number cube 3
num_3_block
```

统一成一个稳定名字，例如：

```text
number_block_3
```

原因：

- YOLO 是闭集分类器，训练时会把不同 class id 当成不同类别。
- synonym label 会把正样本拆散，造成模型在同一个物体上学到互相竞争的类别。
- 下游 planner/executor 需要稳定 label，不能在运行时猜 `number block 3` 和 `number cube 3` 是不是同一个东西。

推荐规则：

```text
全部小写
单词用 underscore
不使用空格
不使用 synonym
不使用临时别名
```

示例：

```text
number_block_1
number_block_2
number_block_3
paper_cup
yellow_bin
```

### 0.2 需要自己写完整标注 GUI 吗？

不需要。

完整标注 GUI 很耗时间，而且现成工具已经支持 bbox、class 管理、多人协作、导入导出、review、版本管理和 model-assisted annotation。建议使用：

```text
CVAT
Roboflow
Label Studio
Ultralytics Platform
```

本 repo 只建议维护这些工程脚本：

```text
frame extraction
dedup
pre-label generation
label check
label visualization
train script
eval script
export script
deployment config
```

也就是说，AgenticLab 代码库负责机器人数据流和训练入口；人工标注交给成熟平台。

### 0.3 最实用的数据采集路线是什么？

建议使用：

```text
camera 拍视频
  -> sparse frame extraction
  -> 去重
  -> 批量预标注
  -> 人工修正
  -> train/val/test split
```

这是 AgenticLab 桌面机器人最实用的路线。原因是机器人真实运行时的数据分布来自 camera viewpoint、桌面光照、物体遮挡、机械臂进入画面、夹爪持物、堆叠关系和失败案例；这些东西很难靠网上数据集覆盖。

## 1. Repo Boundary

当前 repo 不需要先做一个庞大的数据平台。建议边界如下：

```text
train_yolo/
  data_collect.py                  # camera/video frame collection, extraction, dedup helper
  train.py                         # YOLO training entry
  detect.py                        # local prediction / visualization entry
  train.md                         # this workflow
  datasets/
    agenticlab_objects/
      data.yaml
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
  raw/
    videos/
    frames/
    unlabeled/
  runs/
    train/
    predict/
    pseudo_label/
```

部署相关代码仍在：

```text
src/agenticlab_human/perception/detection/yolo_world_detector.py
configs/perception/obj_detector_config.yaml
```

训练好的普通 YOLO 模型部署时使用：

```yaml
model_type: "regular"
```

因为 fine-tuned `yolo26s.pt` 是闭集 detector，不再是 text-prompt open-vocabulary detector。

## 2. Class Taxonomy

先写一个稳定 class list，再开始标注。不要边标边改名字。

示例 class list：

```text
number_block_1
number_block_2
number_block_3
paper_cup
yellow_bin
```

命名建议：

- `number_block_3` 比 `number cube 3` 更适合作为训练 label。
- 如果任务需要区分数字，就把数字写进 class：`number_block_3`。
- 如果任务只需要抓所有数字积木，可以统一为 `number_block`，数字由 OCR/VLM/后处理识别。
- 如果两个物体只有实例差异，但类别和语义一样，不要拆成 `blue_cube_1` / `blue_cube_2`。
- 如果 planner 会说 `pick number block 3`，可以在 planner 到 detector 的映射层把自然语言 alias 统一映射到 `number_block_3`，不要把 alias 放进 YOLO label。

建议维护一个 alias map，而不是维护多个 YOLO class：

```yaml
number_block_3:
  - number block 3
  - number cube 3
  - block with number 3
```

但 `data.yaml` 中只出现：

```yaml
2: number_block_3
```

## 3. Dataset YAML

Ultralytics YOLO detection dataset 使用一个 YAML 描述数据集根目录、train/val/test image 路径和 class names。label 文件是一张图一个 `.txt`，每一行是 normalized bbox。

推荐 `train_yolo/datasets/agenticlab_objects/data.yaml`：

```yaml
path: train_yolo/datasets/agenticlab_objects
train: images/train
val: images/val
test: images/test

names:
  0: number_block_1
  1: number_block_2
  2: number_block_3
  3: paper_cup
  4: yellow_bin
```

也可以写成 list：

```yaml
path: train_yolo/datasets/agenticlab_objects
train: images/train
val: images/val
test: images/test

names:
  - number_block_1
  - number_block_2
  - number_block_3
  - paper_cup
  - yellow_bin
```

注意：

- `class_id` 从 `0` 开始。
- `names` 顺序一旦用于标注和训练，就不要随意调整。
- 新增类别时加在末尾，避免旧 label class id 全部错位。
- 如果删除/重排类别，必须重新转换或检查所有 label。

### 3.1 Optional NDJSON Dataset Format

Ultralytics 也支持 NDJSON。NDJSON 是 Newline Delimited JSON，每一行是一个独立 JSON object。它可以把 dataset metadata、image path、split、image size 和 annotations 放在同一个 `.ndjson` 文件中。

NDJSON 文件结构：

```text
line 1:
  dataset record
  包含 task、dataset name、class_names、version、created_at 等 metadata

line 2+:
  image records
  每行描述一张 image，包括 file/url、width、height、split、annotations
```

AgenticLab detect dataset 的第一行可以写成：

```json
{"type":"dataset","task":"detect","name":"agenticlab_objects","description":"AgenticLab tabletop object detection dataset","class_names":{"0":"number_block_1","1":"number_block_2","2":"number_block_3","3":"paper_cup","4":"yellow_bin"},"version":0,"created_at":"2026-06-02T00:00:00Z","updated_at":"2026-06-02T00:00:00Z"}
```

后续 image record 示例：

```json
{"type":"image","file":"images/train/000001.jpg","width":1280,"height":720,"split":"train","annotations":{"boxes":[[0,0.5375,0.6972,0.0320,0.0736],[3,0.6789,0.7138,0.0656,0.1013]]}}
{"type":"image","file":"images/val/000101.jpg","width":1280,"height":720,"split":"val","annotations":{"boxes":[[4,0.5000,0.5000,0.1200,0.1800]]}}
{"type":"image","file":"images/test/000201.jpg","width":1280,"height":720,"split":"test"}
```

`annotations.boxes` 仍然使用 YOLO detect 的 normalized xywh 格式：

```text
[class_id, x_center, y_center, width, height]
```

训练时可以直接把 `data` 指向 `.ndjson`：

```bash
yolo detect train \
  model=yolo26s.pt \
  data=train_yolo/datasets/agenticlab_objects/agenticlab_objects.ndjson \
  imgsz=960 \
  epochs=150 \
  batch=16 \
  device=0
```

Python：

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
model.train(
    data="train_yolo/datasets/agenticlab_objects/agenticlab_objects.ndjson",
    imgsz=960,
    epochs=150,
)
```

什么时候用 NDJSON：

- 从 Ultralytics Platform 导出 dataset snapshot。
- 想把 metadata、split 和 annotations 放在一个文件里做版本管理。
- 数据集很大，希望 line-by-line streaming。
- 有远程 image URL 或云端训练需求。

什么时候继续用 `data.yaml` + `labels/*.txt`：

- 本地训练和调试。
- 需要兼容 CVAT / Roboflow / Label Studio 常见 YOLO export。
- 想直接检查每张图片对应的 label 文件。

对当前 AgenticLab 第一版数据集，`data.yaml` + YOLO txt 仍然是最直观的路线；NDJSON 可以作为平台导出、云端训练或 dataset snapshot 的补充格式。

## 4. YOLO Label Format

每张图片对应一个同名 `.txt`：

```text
images/train/000001.jpg
labels/train/000001.txt
```

每一行格式：

```text
class_id x_center y_center width height
```

所有 bbox 坐标都归一化到 `[0, 1]`：

```text
0 0.5375 0.6972 0.0320 0.0736
2 0.6789 0.7138 0.0656 0.1013
```

含义：

```text
class_id = data.yaml names 或 NDJSON class_names 中的类别 index
x_center = bbox center x / image_width
y_center = bbox center y / image_height
width = bbox width / image_width
height = bbox height / image_height
```

不要把 pixel coordinate 直接写进 YOLO `.txt`。

Ultralytics 文档说明无目标图片可以没有 label 文件；工程上为了检查方便，也可以保留空 `.txt`。本 repo 建议检查脚本同时接受这两种情况。

## 5. Data Collection

### 5.1 最推荐路线

对 AgenticLab，第一版数据集建议这样做：

```text
1. 用真实 camera 拍 10-30 段短视频。
2. 每段视频覆盖一种任务场景、光照、摆放或遮挡变化。
3. 从视频中按 1-3 FPS sparse extraction。
4. 用 perceptual hash / embedding / manual review 去掉近重复帧。
5. 批量跑预标注。
6. 在 CVAT / Roboflow / Label Studio / Ultralytics Platform 人工修正。
7. 按 session split 成 train/val/test。
```

不要直接把连续视频帧随机切分到 train 和 val。相邻帧太像，会让 val mAP 虚高。

### 5.2 Video To Sparse Frames

示例：

```bash
mkdir -p train_yolo/raw/frames/session_001
ffmpeg -i train_yolo/raw/videos/session_001.mp4 \
  -vf fps=2 \
  train_yolo/raw/frames/session_001/%06d.jpg
```

如果动作变化很慢，用 `fps=1`。如果夹爪动作、遮挡变化很快，用 `fps=3`。

### 5.3 Capture Metadata

建议每个 session 保存 metadata：

```json
{
  "session": "20260602_session_001",
  "camera": "orbbec_femto_bolt",
  "resolution": [1280, 720],
  "scene": "agenticlab_tabletop",
  "lighting": "lab_normal",
  "objects": ["number_block_1", "number_block_2", "number_block_3"],
  "notes": "blocks stacked and partially occluded"
}
```

训练 YOLO 只需要 RGB，但 metadata 对后续 error analysis 很有价值。

### 5.4 Coverage Checklist

第一版数据集至少覆盖：

- 单个物体居中。
- 多个物体同时出现。
- 同类多个实例同时出现。
- 小物体、边缘位置、远处位置。
- 部分遮挡。
- 堆叠、贴近、相互接触。
- 夹爪进入画面。
- 物体被夹起或刚放下。
- 空桌面或无目标 negative scene。
- 真实失败案例。
- 不同光照、背景和桌面区域。

推荐第一阶段规模：

```text
200-300 clean labeled images
train/val/test split by session
每个关键 class 至少 30-50 个 bbox
```

## 6. Preprocess And Dedup

预处理目标不是“把图片变漂亮”，而是让训练数据稳定、可检查、不过度重复。

建议步骤：

```text
raw frames
  -> remove corrupt images
  -> optional resize only for preview
  -> dedup near-identical frames
  -> group by session
  -> split train/val/test by session
```

注意：

- 训练前不需要手动把图片 resize 到 `imgsz`；Ultralytics training 会处理。
- 不要把 near duplicate 同时放进 train 和 val。
- 不要过度清洗，只保留“完美照片”；机器人真实运行会有模糊、遮挡、反光和夹爪。

## 7. Pre-labeling

预标注的目的只是减少人工画框时间，不是替代人工检查。

### 7.1 用已有 YOLO/YOLOE/YOLO-World 预标注

如果当前模型能识别这些对象，可以先生成预测图：

```bash
python src/agenticlab_human/perception/detection/yolo_world_detector.py \
  --image train_yolo/raw/frames/session_001/000001.jpg \
  --classes number_block_1 number_block_2 number_block_3 \
  --model-path yoloe-26x-seg.pt \
  --model-type yoloe \
  --conf 0.35 \
  --imgsz 960
```

对于已经 fine-tune 的闭集 YOLO：

```bash
python src/agenticlab_human/perception/detection/yolo_world_detector.py \
  --image train_yolo/raw/frames/session_001/000001.jpg \
  --classes number_block_1 number_block_2 number_block_3 \
  --model-path train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  --model-type regular \
  --conf 0.25 \
  --imgsz 960
```

### 7.2 生成 YOLO txt 建议标签

对一批 unlabeled images：

```bash
yolo detect predict \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  source=train_yolo/raw/unlabeled \
  imgsz=960 \
  conf=0.50 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=train_yolo/runs/prelabel \
  name=v0_teacher
```

导出的 `.txt` 可以作为人工修正起点。合并进正式 dataset 前必须 review。

### 7.3 Ultralytics `auto_annotate`

Ultralytics 的 `auto_annotate` 是 YOLO detection model + SAM 的辅助标注流程，官方示例：

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="train_yolo/raw/frames/session_001",
    det_model="yolo26x.pt",
    sam_model="sam_b.pt",
)
```

注意：这个流程主要用于生成 segmentation/mask annotation。若最终训练 detection YOLO，可以在标注平台里导出 detection bbox，或把 mask 转为 bbox 后再导出 YOLO detect 格式。不要把未经检查的自动 mask/box 直接当 ground truth。

## 8. Human Correction

推荐工具：

```text
CVAT
Roboflow
Label Studio
Ultralytics Platform
```

人工修正重点：

- 删除重复框。
- 补上漏标的小物体和遮挡物体。
- 统一 synonym 到 canonical class。
- 调整过松或过紧的 bbox。
- 检查 class id 是否与 `data.yaml` names 或 NDJSON `class_names` 一致。
- 给 negative images 保持无目标标签。

本 repo 不建议实现完整 GUI。最多实现：

```text
label_check.py       # 检查文件缺失、class id 越界、bbox 越界
visualize_labels.py  # 把 label 画回图片做抽查
dedup.py             # 去重
split_dataset.py     # 按 session split
```

## 9. Train With `yolo26s.pt`

### 9.1 CLI

正式训练示例：

```bash
yolo detect train \
  model=yolo26s.pt \
  data=train_yolo/datasets/agenticlab_objects/data.yaml \
  imgsz=960 \
  epochs=150 \
  batch=16 \
  device=0 \
  workers=8 \
  project=train_yolo/runs/train \
  name=agenticlab_objects_yolo26s \
  patience=30 \
  close_mosaic=10 \
  cache=False
```

如果使用 NDJSON，把 `data` 换成 `.ndjson` 路径即可：

```bash
yolo detect train \
  model=yolo26s.pt \
  data=train_yolo/datasets/agenticlab_objects/agenticlab_objects.ndjson \
  imgsz=960 \
  epochs=150 \
  batch=16 \
  device=0 \
  workers=8 \
  project=train_yolo/runs/train \
  name=agenticlab_objects_yolo26s_ndjson \
  patience=30 \
  close_mosaic=10
```

CPU smoke test 只用于检查 dataset 格式：

```bash
yolo detect train \
  model=yolo26s.pt \
  data=train_yolo/datasets/agenticlab_objects/data.yaml \
  imgsz=640 \
  epochs=1 \
  batch=2 \
  device=cpu \
  project=train_yolo/runs/train \
  name=smoke_cpu
```

CPU 不适合正式训练。

### 9.2 Python Script

建议把 `train_yolo/train.py` 写成：

```python
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo26s.pt")
    model.train(
        data="train_yolo/datasets/agenticlab_objects/data.yaml",
        imgsz=960,
        epochs=150,
        batch=16,
        workers=8,
        device=0,
        project="train_yolo/runs/train",
        name="agenticlab_objects_yolo26s",
        pretrained=True,
        patience=30,
        close_mosaic=10,
        cache=False,
        single_cls=False,
    )


if __name__ == "__main__":
    main()
```

若训练 NDJSON dataset，只需要改 `data`：

```python
data="train_yolo/datasets/agenticlab_objects/agenticlab_objects.ndjson"
```

运行：

```bash
python train_yolo/train.py
```

### 9.3 Small Dataset Recipe

如果只有 `200-300` 张 clean labeled images，优先用 pretrained checkpoint，而不是从 scratch 训练。

保守 two-stage fine-tuning：

```python
from ultralytics import YOLO


DATA = "train_yolo/datasets/agenticlab_objects/data.yaml"


def main() -> None:
    model = YOLO("yolo26s.pt")

    model.train(
        data=DATA,
        imgsz=960,
        epochs=50,
        batch=16,
        device=0,
        freeze=10,
        lr0=1e-4,
        project="train_yolo/runs/train",
        name="stage1_frozen",
        patience=15,
    )

    model = YOLO("train_yolo/runs/train/stage1_frozen/weights/best.pt")
    model.train(
        data=DATA,
        imgsz=960,
        epochs=100,
        batch=16,
        device=0,
        lr0=5e-5,
        project="train_yolo/runs/train",
        name="stage2_unfrozen",
        patience=25,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
```

为什么这样做：

- pretrained weights 对小数据集很重要。
- 第一阶段 freeze backbone 可以降低过拟合风险。
- 第二阶段全量微调让模型适应真实 AgenticLab 视角。
- `close_mosaic=10` 让最后若干 epoch 更接近真实图片分布。

## 10. Augmentation

AgenticLab tabletop 推荐：

```python
model.train(
    data="train_yolo/datasets/agenticlab_objects/data.yaml",
    imgsz=960,
    epochs=150,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.35,
    degrees=5,
    translate=0.08,
    scale=0.4,
    fliplr=0.0,
    mosaic=0.8,
    mixup=0.05,
    copy_paste=0.1,
    close_mosaic=10,
)
```

避免：

- 过大旋转，让桌面和重力方向不真实。
- 过强颜色增强，让颜色类物体变得语义不清。
- 过度 crop，导致保留完整 label 但物体已严重缺失。
- train/val 使用几乎相同的连续视频帧。

## 11. Evaluation

### 11.1 Ultralytics Val

```bash
yolo detect val \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  data=train_yolo/datasets/agenticlab_objects/data.yaml \
  imgsz=960 \
  device=0
```

关注：

```text
mAP50
mAP50-95
precision
recall
per-class AP
confusion matrix
val_batch*_pred.jpg
```

输出通常在：

```text
train_yolo/runs/train/<experiment>/
  results.csv
  results.png
  confusion_matrix.png
  val_batch*_pred.jpg
  weights/best.pt
  weights/last.pt
```

### 11.2 Robot-oriented Check

mAP 好不等于机器人能抓。还要检查：

- target object 是否漏检。
- bbox center 是否足够接近物体中心。
- bbox 是否覆盖可抓区域，而不是大面积包含背景。
- support object 和 target object 是否混淆。
- close objects 是否出现 duplicate boxes。

使用 repo detector 做 task-level smoke test：

```bash
python src/agenticlab_human/perception/detection/yolo_world_detector.py \
  --image data/data_for_test/task_parser/04_stack1_color.png \
  --classes number_block_1 number_block_2 number_block_3 \
  --model-path train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  --model-type regular \
  --conf 0.25 \
  --imgsz 960
```

预期输出形态：

```json
{
  "success": true,
  "num_objects": 3,
  "save_dir": "output/perception/yolo_world/<timestamp>/<labels>_result"
}
```

实际 `num_objects` 取决于测试图片。

## 12. Detection Script

建议 `train_yolo/detect.py`：

```python
from ultralytics import YOLO


def main() -> None:
    model = YOLO("train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt")
    results = model.predict(
        source="data/data_for_test/task_parser/04_stack1_color.png",
        imgsz=960,
        conf=0.25,
        save=True,
        project="train_yolo/runs/predict",
        name="agenticlab_objects",
    )
    print(results[0].boxes)


if __name__ == "__main__":
    main()
```

运行：

```bash
python train_yolo/detect.py
```

## 13. Export ONNX And TensorRT

### 13.1 ONNX

ONNX 适合跨平台部署、CPU/GPU runtime、调试和模型交付：

```bash
yolo export \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  format=onnx \
  imgsz=960 \
  simplify=True
```

Python：

```python
from ultralytics import YOLO

model = YOLO("train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt")
model.export(format="onnx", imgsz=960, simplify=True)
```

### 13.2 TensorRT

TensorRT engine 要在目标 GPU / CUDA / TensorRT 环境上导出：

```bash
yolo export \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  format=engine \
  imgsz=960 \
  device=0 \
  half=True
```

INT8 TensorRT 需要 calibration data，建议显式传入同一个 dataset YAML：

```bash
yolo export \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  format=engine \
  imgsz=960 \
  device=0 \
  int8=True \
  data=train_yolo/datasets/agenticlab_objects/data.yaml
```

保留 `.pt` 作为开发和回归测试基准。ONNX/TensorRT 只在 `.pt` 通过 validation 和 robot-oriented check 后再进入部署。

## 14. Deploy To AgenticLab

更新 `configs/perception/obj_detector_config.yaml`：

```yaml
YoloWorldDetector:
  model_path: "train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt"
  model_type: "regular"
  confidence: 0.25
  image_size: 960
  output_dir: "output/perception/yolo_world"
```

测试：

```bash
python src/agenticlab_human/perception/detection/yolo_world_detector.py \
  --image data/data_for_test/task_parser/04_stack1_color.png \
  --classes number_block_1 number_block_2 number_block_3 \
  --model-path train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  --model-type regular \
  --conf 0.25 \
  --imgsz 960
```

注意：

- 普通 YOLO 是闭集模型，`--classes` 只能请求或过滤模型 `names` 里存在的 class。
- 如果报 `unavailable_requested_classes`，说明 planner/requested class 和 `data.yaml` canonical class 不一致。
- 不要在部署时临时使用 synonym；应在上游 alias map 统一到 canonical class。

## 15. Active Learning

部署后持续收集 hard cases：

```text
output/perception/failure_cases/
  missed_object/
  duplicate_box/
  wrong_class/
  bad_box_for_grasp/
  low_confidence/
  new_scene/
```

优先送回标注的数据：

- robot 因漏检无法执行任务。
- bbox center 明显偏离物体中心。
- 相邻物体产生 duplicate / merged boxes。
- target object 和 support object 混淆。
- 新光照、新桌面、新 camera angle。
- 新物体组合或新遮挡关系。

推荐循环：

```text
v0:
  200-300 clean images
  train from yolo26s.pt
  deploy in dry-run perception mode

active learning batch:
  collect 50-100 hard/uncertain images
  correct labels
  keep validation set fixed

v1:
  train from v0 best.pt or yolo26s.pt
  compare v0 vs v1 on same val/test
  deploy only if robot-oriented check improves
```

## 16. Pseudo-labeling

Pseudo-labeling 用于扩展“容易样本”，不是替代人工标注。

生成 teacher prediction：

```bash
yolo detect predict \
  model=train_yolo/runs/train/agenticlab_objects_yolo26s/weights/best.pt \
  source=train_yolo/raw/unlabeled \
  imgsz=960 \
  conf=0.65 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=train_yolo/runs/pseudo_label \
  name=v0_teacher
```

合并规则：

- 只保留高置信度、肉眼确认正确的 pseudo-label。
- hard cases 仍然人工标注，不依赖 pseudo-label。
- pseudo-labeled images 不要压过人工标注数据，避免模型复制 teacher 的系统性错误。
- 每次合并后重新跑 `label_check` 和 visualization 抽查。

## 17. COCO Or Public Dataset To YOLO

如果补充外部数据，必须先统一 class taxonomy。不要把外部 dataset 的 label 原样混进来。

流程：

```text
COCO/public dataset
  -> filter relevant classes
  -> rename labels to AgenticLab canonical class
  -> convert to YOLO txt or Ultralytics NDJSON
  -> visualize labels
  -> merge only if camera/domain gap is acceptable
```

Ultralytics 文档支持 YOLO format dataset 和 NDJSON dataset，也提供 COCO-to-YOLO 的转换思路。对 AgenticLab 来说，外部数据只能作为补充；真实 camera 数据仍然优先。

## 18. Common Problems

### 18.1 No detections after training

检查：

- `data.yaml` 或 `.ndjson` 路径是否正确。
- `names` 顺序是否和 label `class_id` 一致。
- label bbox 是否 normalized，而不是 pixel coordinate。
- image 和 label stem 是否匹配。
- validation set 是否为空。
- predict `conf` 是否太高。
- 部署时 `model_type` 是否为 `regular`。

### 18.2 Validation mAP 很高，但机器人效果差

常见原因：

- train/val 有 near duplicate。
- val 没有按 session split。
- 数据缺少真实 camera viewpoint。
- 数据缺少遮挡、堆叠、夹爪、失败案例。
- bbox 对 mAP 足够，但对 grasp center 不够稳定。

### 18.3 同一个物体预测出多个类别

常见原因：

- 训练集中存在 synonym label。
- class taxonomy 太细，视觉上不可区分。
- 人工标注 class id 不一致。

修复：

```text
统一 canonical class
重写 alias map
清理旧 label
重新训练
```

### 18.4 数字积木 `number_block_3` 和其他数字混淆

尝试：

- 增加正面、侧面、远近、遮挡样本。
- 增加每个数字的均衡样本。
- 如果数字太小，训练 `number_block` detector，再由 OCR/VLM 做数字识别。
- 提高输入 `imgsz`，例如 `960` 或 `1280`，但要评估速度。

## 19. Recommended First Milestone

第一版 milestone：

```text
Dataset:
  200-300 labeled AgenticLab camera RGB images
  session-based train/val/test split
  no synonym labels
  no near duplicate leakage

Classes:
  number_block_1
  number_block_2
  number_block_3
  paper_cup
  yellow_bin

Model:
  yolo26s.pt

Training:
  imgsz=960
  epochs=150
  batch=16 if GPU memory allows
  patience=30
  close_mosaic=10

Acceptance:
  stable bbox on held-out tabletop scenes
  no missed target object in dry-run action sequences
  bbox center good enough for grasp assignment
  no planner/requested class mismatch
```

完成后再扩展类别、加更多 failure cases，并考虑更大的 YOLO checkpoint。
