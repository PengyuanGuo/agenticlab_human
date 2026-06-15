# Code Inventory

## Core
- core/action_sequence.py: Planner → Executor contract.
- execution/action_backend.py: Robot backend interface.
- planning/task_parser.py: Converts task + scene into PDDL/action sequence.

## Runtime Backends
- execution/x5_backend.py: X5 robot execution.
- execution/flexiv_backend.py: Flexiv execution.

### X5 Client/Server
- `execution/robot/x5/contracts.py`: FastAPI/Pydantic command models, robot state
  models, and versioned RGB-D NPZ encoding/decoding.
- `execution/robot/x5/camera.py`: `RGBDCamera` protocol and the Orbbec adapter
  that returns aligned RGB, millimeter depth, runtime intrinsics, and frame
  timestamps.
- `execution/robot/x5/x5_controller.py`: `X5Controller` protocol and the real
  xapi controller. Owns tool-frame setup, state conversion, motion dispatch,
  and server-side motion safety checks.
- `execution/robot/x5/mock_controller.py`: Deterministic dual-arm mock
  controller used by HTTP tests.
- `execution/robot/x5/gripper_controller.py`: Dahuan serial register driver,
  server-side normalized gripper service, and mock gripper service.
- `execution/robot/x5/conversion.py`: Shared angle, quaternion, rotation-vector,
  X5 pose, and SE(3) conversions.
- `execution/robot/x5/server.py`: FastAPI application, hardware lifecycle,
  per-device single-thread executors, health/capture/robot endpoints, and YAML
  config assembly.
- `execution/robot/x5/client.py`: Synchronous `X5HTTPClient`, RGB-D artifact
  saving/preview, and low-level robot command helpers.
- `execution/robot/x5/x5_remote_backend.py`: Planner-facing remote X5 action
  backend. Converts AnyGrasp poses to world-frame TCP poses and executes the
  pick/place sequence through `X5HTTPClient`.
- `configs/robot/x5_config.yaml`: Server hardware, safety limits, tool frame,
  home/check poses, gripper, and client-side action-backend settings.
- `configs/execution/x5_pipeline.yaml`: End-to-end X5 execution pipeline,
  detector, grasp, and placement settings.
- `tests/test_x5_http.py`: In-process mock tests for HTTP health, RGB-D
  round-trip, robot/gripper commands, validation, and saved capture artifacts.
- `tests/test_x5_controller.py`: xapi boundary tests for public-unit state,
  tool/point conversion, safety limits, stop ordering, and gripper mapping.
- `tests/test_x5_remote_backend.py`: Pick/place transform, trajectory ordering,
  home segmentation, gripper, and failure-stop tests.

## Perception
- perception/yolo_detector.py: Object bbox detection.
- grasp/graspnet_client.py: HTTP client to AnyGrasp/GraspNet service.

## Unclear / Candidate for deletion
- xxx.py: Seems duplicated with yyy.py.
- old_demo_xxx.py: One-off test.


#  The rest of the content is borrowed from Tiptop/Claude.md, do not processed yet.

## Coding Style and Principles

### Function-based Design
- **Prefer functions over classes**: Use standalone functions and functional programming patterns where possible
- Only use classes when managing stateful operations 
- Classes that act as callable function containers should implement `__call__()` to maintain functional interface

### Documentation
- Use concise single-line docstrings for simple functions
- Only add detailed docstrings when the function behavior is non-trivial or requires explanation
- Avoid redundant documentation that simply restates the function name

### Code Organization
- Break complex logic into focused, single-purpose functions
- Keep functions cohesive - extract logical units like transformation computations or data processing
- Use descriptive function names that clearly indicate purpose 
- Avoid over-fragmenting code into too many tiny functions

### Naming Conventions
- Use descriptive variable names that indicate transformations: 
- Follow the pattern `target_from_source` for transformation matrices
- Use `_fn` suffix for higher-order functions (e.g., `grasp_to_mat4x4_fn`)

### Type Hints
- Use type hints for function parameters and return values
- Leverage  for tensor dimensions 
- Define TypedDict classes for complex return structures 

### Error Handling
- Validate inputs at the start of functions with clear error messages
- Use `raise ValueError` or `raise RuntimeError` with descriptive messages
- Include context in error messages (e.g., which object/parameter caused the issue)

### Imports
- Group imports: standard library, third-party, local imports
- Use absolute imports from  package root
- Avoid wildcard imports
- Avoid local imports inside functions for standard modules; import at module level

### String Formatting
- Always use f-strings for string formatting; avoid `%s` or `.format()`


## Working Style

- **Discuss before implementing** — When the approach isn't obvious (e.g., where to put docs, how to structure config), talk through options before writing code
- **Comments and docstrings must be accurate** — Don't write comments that describe implementation details irrelevant to the reader or are vaguely wrong. If a comment doesn't add real information, drop it
- **Don't add dead code paths** — If every case goes down the same branch, don't add the other branch "just in case." Keep what's tested, remove what isn't

## Documentation Style Guide

When working on documentation:
- Be concise without making the writing feel unnatural
- Handle edge cases appropriately but not exhaustively
- Avoid excessive verbosity
- Use MyST-Parser Markdown features (colon fences, admonitions, etc.)
- Follow the existing black & white theme aesthetic
- API documentation will be auto-generated from docstrings once source code is added

## Key Concepts
