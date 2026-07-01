# AgenticLab Human Main Tutorial

## Goals
After reading this tutorial, you should be able to:
1. Understand the code structure of AgenticLab on the H1 humanoid robot upper body.
2. Understand the purpose of the main modules.
3. Install the main modules: voice-to-text, text-to-plan, perception, GraspNet, and X5 communication dependencies.
4. Run the voice-to-text generation script.
5. Use remote camera capture to collect RGB-D images.
6. Run GraspNet in a local environment to generate grasp poses.
7. Run a simple X5 arm control script through the FastAPI framework.
8. Run a simple pick-and-place pipeline on the H1 humanoid robot.

## Main Modules
1. Voice-to-text and text-to-plan generation.
2. Planning with VLMs and Fast Downward.
3. X5 robot control and communication framework.
4. Camera perception and utilities.
5. Grasp generation and GraspNet environment setup.
6. YOLO detection and fine-tuning with new object classes.

## Main Code Architecture

## Module Documents
1. Voice-to-text and voice-to-plan generation. [voice/voice_module.md](voice/voice_module.md).
2. Planning with VLMs and Fast Downward.
3. X5 robot control and communication framework: [X5/client_server.md](X5/client_server.md).
4. Camera perception and utilities. [perception/camera_usage.md](perception/camera_usage.md).
5. Grasp generation and GraspNet environment setup: [grasp/grasp_client_server.md](grasp/grasp_client_server.md).
6. YOLO detection and fine-tuning: [perception/yolo_train.md](perception/yolo_train.md).
7. Pipeline execution: [pipeline/execution_without_planning.md](pipeline/execution_without_planning.md).

## Installation
### 1. Install the AgenticLab repository
```bash
conda create -n agenticlab_human python=3.10
conda activate agenticlab_human
cd /home/agenticlab/Project/agenticlab_human # switch to your own project root
python -m pip install -e .
```
### 2. Install voice-to-text dependencies
```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[voice]"
```
After installing the voice-to-text module, verify the setup as Basic Operation 1.
### 3. Install VLM calling and Fast Downward dependencies
```bash
sudo apt install cmake g++ make python3
```
To build the planner, run:
```bash
cd /home/agenticlab/Project/agenticlab_human/third_party/downward # switch to your own project root
./build.py
```
To test your build use:
```bash
./fast-downward.py misc/tests/benchmarks/miconic/s1-0.pddl --search "astar(lmcut())"
```
### 4. Install camera dependencies
Recommend online installation:
```bash
pip install --upgrade pyorbbecsdk2   # on server pc
```
For off-line installation: please refer to 
https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html#method-2-offline-wheel-installation

Verifying on the installed pc: 
```bash
python src\agenticlab_human\perception\camera\orbbec_capture.py
```
### 5. Install FastAPI dependencies
```bash
python -m pip install -e ".[x5]"     # both on client and server pc
```
After installation of FastAPI on both PC, you can verify by Basic Operation 2
### 6. Install GraspNet dependencies

Since GraspNet has heavy dependencies and complicated installation of Minkowski Engine, please refer to the installation guide.

[grasp/installation_guide_of_anygrasp (open-sourced).md](grasp/installation_guide_of_anygrasp (open-sourced).md)
### 7. Install YOLO detector
```bash
pip install ultralytics
```
For how to collect data, annotate, and finetune YOLO, please refer to [perception/yolo_train.md](perception/yolo_train.md)

## Basic Operation
### 1. Run the voice-to-text generation script （after connect to DJI microphone）
```bash
python -m agenticlab_human.voice.demo_speech_to_text
```
```bash
export OPENAI_API_KEY="you own choice of VLM key"
```
voice to plan generation example, e.g. speak to mic: 把桌上的苹果放到蓝色盒子里
```bash
python -m agenticlab_human.planning.voice_to_planner --image-path data/data_for_test/task_parser/01_sort1_color.png
```
### 2. Capture RGB-D images from the remote camera
```bash
# On the server PC(windows, xiaotuo)
python src\agenticlab_human\execution\robot\x5\server.py --config configs\robot\x5_config.yaml
```
```bash
# On the client PC, capture a single frame of rgb and depth image
python -m agenticlab_human.execution.robot.x5.client \
  --server-url http://192.168.1.15:8000                # server pc ip
  --save-dir output/x5_http_captures/capture1 \         # camera capture save directory
  --preview
```
### 3. Generate grasp poses with GraspNet in the local environment
```bash
# Open another terminal
conda activate flexiv        # activate the env with GraspNet

OMP_NUM_THREADS=12 PYTHONPATH=src python -m  \
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

```bash
# In agentic_human env, to verify the usage of GraspNet
python -m agenticlab_human.perception.grasping.client   --server-url http://127.0.0.1:8010   --rgb-path output/x5_http_captures/orbbec-0000093038_rgb.png  --depth-path output/x5_http_captures/orbbec-0000093038_depth_mm.png    --bbox 640 476 836 690   --mask-offset-px 20   --object-label red_bin
```

### 4. Detect an object on the captured image using fineYOLO
```bash
python src/agenticlab_human/perception/detection/yolo.py  --image output/x5_http_captures/orbbec-0000093038_rgb.png  --output-dir output/perception/yolo_test --conf 0.25 --model-path train_yolo/runs/agenticlab_red_bin_002_yolo26s/weights/best.pt
```

### 5. Run a simple pick-and-place script on X5
```bash
### run this on local pc
python -m agenticlab_human.execution.pipeline pipeline   --object number_block_2   --target red_bin   --config configs/execution/x5_pipeline.yaml   --execute
```

### 6. Run pick-place-pour execution loop on X5
```bash

```
