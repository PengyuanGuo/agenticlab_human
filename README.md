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