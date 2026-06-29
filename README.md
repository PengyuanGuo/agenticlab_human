# AgenticLab Human / X5

## What this repo does

Voice/text task → planner → ActionSequence → perception/grasp cache → robot backend execution. (mainly on X5 arm)


## Main entrypoints

- Plan only:
- Execute existing plan:
- Dry-run execution:
- X5 execution:
- Flexiv execution:



## Most Used Commands



- run voice pipeline
CLI to run:
- run explicit pick-place
```bash
python -m agenticlab_human.execution.pipeline pipeline \
  --object number_block_3 \
  --target yellow_bin \
  --config configs/execution/x5_pipeline.yaml \
  --execute
```
- start X5 server
```bash
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_config.yaml
```
- start grasp server
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
- Start X5 client
```bash
python -m agenticlab_human.execution.robot.x5.client \
  --home \
  --server-url http://192.168.1.15:8000
```
- Control Dahuan Gripper on server PC
```bash
python -m agenticlab_human.execution.robot.x5.gripper_controller --open
```


## For more detailed tutorial, please check /agenticlab_human/docs/AgenticLab_human_main_tutorial.md

## Current Status
X5 humanoid upper body can execute pick-place-pour style demos.
Known limitation: some grasp poses are unreachable due to arm soft limits / IK feasibility.