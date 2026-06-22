# agenticlab_human

## What this repo does
Voice/text task → planner → ActionSequence → perception/grasp cache → robot backend execution.

## Current main pipeline
Planner produces ActionSequence.
ActionExecutor consumes ActionSequence.
ActionBackend hides robot-specific control.

## Main entrypoints
- Plan only:
- Execute existing plan:
- Dry-run execution:
- X5 execution:
- Flexiv execution:

## Core contracts
- ActionSequence
- ActionBackend
- ExecutionContext
- PerceptionBackend
- GraspBackend

## Hardware status
- Flexiv:
- X5:
- Orbbec:
- AnyGrasp/GraspNet:

CLI to run:
```bash
python -m agenticlab_human.execution.pipeline pipeline \
  --object number_block_3 \
  --target yellow_bin \
  --config configs/execution/x5_pipeline.yaml \
  --execute
```

```bash
python -m agenticlab_human.execution.robot.x5.server \
  --config configs/robot/x5_config.yaml
```

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

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --home \
  --server-url http://192.168.1.15:8000
```

```bash
python -m agenticlab_human.execution.robot.x5.client \
  --home \
  --server-url http://192.168.1.15:8000
```

```bash
python -m agenticlab_human.execution.robot.x5.gripper_controller --open
```