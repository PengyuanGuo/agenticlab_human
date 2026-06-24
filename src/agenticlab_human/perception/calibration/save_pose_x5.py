"""Interactively save X5 TCP poses for hand-eye calibration.

The CSV format matches ``save_poses_flexiv.py``:
    pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w

Joint targets are entered in degrees for operator convenience. The X5 HTTP
contract continues to use radians, meters, and quaternions in xyzw order.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from agenticlab_human.execution.robot.x5.client import X5HTTPClient


CSV_HEADER = [
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
]


def _find_repo_root(start: Path = Path(__file__).resolve()) -> Path:
    for candidate in (start.parent, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not find repository root containing pyproject.toml")


DEFAULT_CONFIG_PATH = _find_repo_root() / "configs" / "robot" / "x5_config.yaml"


def save_pose_to_csv(pose_data: Sequence[float], filename: str | Path) -> None:
    values = [float(value) for value in pose_data]
    if len(values) != 7:
        raise ValueError("TCP pose must contain [x, y, z, qx, qy, qz, qw]")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("TCP pose contains a non-finite value")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(values)

    print(f"Pose saved to {output_path}: {[f'{value:.6f}' for value in values]}")


def load_home_joints_deg(config_path: str | Path, arm: str) -> list[float]:
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    robot_config = config.get("robot", {})
    arm_config = robot_config.get(arm, {})
    values = arm_config.get("home_joints_deg", robot_config.get("home_joints_deg"))
    if values is None:
        raise ValueError(f"home_joints_deg is not configured for arm '{arm}'")

    joints = [float(value) for value in values]
    if len(joints) < 7:
        raise ValueError("home_joints_deg must contain at least 7 values")
    return joints[:7]


def load_joint_limits_deg(config_path: str | Path) -> list[tuple[float, float]] | None:
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    values = config.get("robot", {}).get("joint_limits_deg")
    if values is None:
        return None
    if len(values) != 7 or any(len(limit) != 2 for limit in values):
        raise ValueError("joint_limits_deg must contain 7 [lower, upper] pairs")

    limits = [(float(limit[0]), float(limit[1])) for limit in values]
    if any(lower >= upper for lower, upper in limits):
        raise ValueError("each joint lower limit must be less than its upper limit")
    return limits


def parse_joint_target_deg(text: str) -> list[float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 7:
        raise ValueError("enter exactly 7 joint angles: j1 j2 j3 j4 j5 j6 j7")
    values = [float(part) for part in parts]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("joint angles must be finite numbers")
    return values


def move_to_joint_target_deg(
    client: Any,
    arm: str,
    target_deg: Sequence[float],
    *,
    speed_ratio: float,
    max_step_deg: float,
    joint_limits_deg: Sequence[Sequence[float]] | None = None,
) -> Any:
    """Move through small joint-space waypoints accepted by the X5 server."""

    target = [float(value) for value in target_deg]
    if len(target) != 7:
        raise ValueError("joint target must contain exactly 7 values")
    if max_step_deg <= 0.0:
        raise ValueError("max_step_deg must be positive")
    if joint_limits_deg is not None:
        if len(joint_limits_deg) != 7:
            raise ValueError("joint_limits_deg must contain exactly 7 pairs")
        for index, (value, limit) in enumerate(
            zip(target, joint_limits_deg, strict=True),
            start=1,
        ):
            lower, upper = float(limit[0]), float(limit[1])
            if value < lower or value > upper:
                raise ValueError(
                    f"joint {index} target {value:.3f} deg is outside "
                    f"configured limit [{lower:.3f}, {upper:.3f}] deg"
                )

    state = client.get_state(arm).state_after.arms[arm]
    start_deg = [math.degrees(value) for value in state.joints_rad]
    max_delta = max(
        abs(target_value - start_value)
        for start_value, target_value in zip(start_deg, target, strict=True)
    )
    step_count = max(1, math.ceil(max_delta / max_step_deg))

    for step_index in range(1, step_count + 1):
        fraction = step_index / step_count
        waypoint_deg = [
            start_value + (target_value - start_value) * fraction
            for start_value, target_value in zip(start_deg, target, strict=True)
        ]
        result = client.move_joints(
            arm,
            [math.radians(value) for value in waypoint_deg],
            speed_ratio=speed_ratio,
            wait=True,
        )
        if not result.success:
            raise RuntimeError(result.error or "X5 server rejected move_joints")
        state = result.state_after.arms[arm]
        print(
            f"  move step {step_index}/{step_count}: "
            + " ".join(f"{value:.2f}" for value in waypoint_deg)
        )
    return state


def print_arm_state(state: Any) -> None:
    joints_deg = [math.degrees(value) for value in state.joints_rad]
    tcp = [float(value) for value in state.tcp_pose_xyzw]
    print("Joint deg [j1..j7]: " + " ".join(f"{value:.3f}" for value in joints_deg))
    print(
        "TCP [x y z qx qy qz qw]: "
        + " ".join(f"{value:.6f}" for value in tcp)
    )


def _save_current_pose(client: X5HTTPClient, arm: str, output_path: Path) -> None:
    state = client.get_state(arm).state_after.arms[arm]
    if state.moving:
        raise RuntimeError(f"{arm} arm is still moving; pose was not saved")

    tcp = [float(value) for value in state.tcp_pose_xyzw]
    quaternion_norm = math.sqrt(sum(value * value for value in tcp[3:7]))
    if quaternion_norm <= 1e-8:
        raise ValueError("TCP quaternion has zero norm")
    if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-3):
        print(f"Warning: TCP quaternion norm is {quaternion_norm:.6f}")

    save_pose_to_csv(tcp, output_path)


def _confirm_home(home_joints_deg: Sequence[float], arm: str) -> bool:
    print(
        f"Home target for {arm} arm [deg]: "
        + " ".join(f"{value:.2f}" for value in home_joints_deg)
    )
    answer = input("Press Enter to move home, or type q to quit > ").strip().lower()
    return answer != "q"


def run_interactive_session(
    client: X5HTTPClient,
    *,
    arm: str,
    home_joints_deg: Sequence[float],
    output_path: Path,
    speed_ratio: float,
    max_step_deg: float,
    joint_limits_deg: Sequence[Sequence[float]] | None,
    skip_home: bool,
) -> int:
    health = client.health()
    if not health.robot.ready:
        raise RuntimeError(f"X5 robot backend is not ready: {health.robot.detail}")
    print(
        f"Connected to {client.base_url}: "
        f"robot={health.robot.backend}, camera={health.camera.backend}"
    )

    state_response = client.get_state(arm)
    if arm not in state_response.state_after.arms:
        raise ValueError(f"arm '{arm}' is not available on the X5 server")
    print_arm_state(state_response.state_after.arms[arm])

    if not skip_home:
        if not _confirm_home(home_joints_deg, arm):
            return 0
        print("Moving to home...")
        state = move_to_joint_target_deg(
            client,
            arm,
            home_joints_deg,
            speed_ratio=speed_ratio,
            max_step_deg=max_step_deg,
            joint_limits_deg=joint_limits_deg,
        )
        print("Home motion complete.")
        print_arm_state(state)

    print("\n--- Controls ---")
    print("Enter 7 joint angles in degrees to move: j1 j2 j3 j4 j5 j6 j7")
    print("Enter or s = save current TCP pose")
    print("h = move home, p = print state, q = quit")
    print("----------------")
    print(f"Pose data will be saved to: {output_path}\n")

    pose_count = 0
    while True:
        command = input(f"[{pose_count} saved] joints / Enter=save > ").strip()
        lowered = command.lower()
        if lowered == "q":
            break
        if lowered in ("", "s"):
            _save_current_pose(client, arm, output_path)
            pose_count += 1
            continue
        if lowered == "p":
            print_arm_state(client.get_state(arm).state_after.arms[arm])
            continue
        if lowered == "h":
            state = move_to_joint_target_deg(
                client,
                arm,
                home_joints_deg,
                speed_ratio=speed_ratio,
                max_step_deg=max_step_deg,
                joint_limits_deg=joint_limits_deg,
            )
            print_arm_state(state)
            continue

        try:
            target_deg = parse_joint_target_deg(command)
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            continue

        print("Moving to joint target...")
        try:
            state = move_to_joint_target_deg(
                client,
                arm,
                target_deg,
                speed_ratio=speed_ratio,
                max_step_deg=max_step_deg,
                joint_limits_deg=joint_limits_deg,
            )
        except ValueError as exc:
            print(f"Invalid target: {exc}")
            continue
        print("Motion complete. Press Enter to save this pose.")
        print_arm_state(state)

    print(f"Done. {pose_count} pose(s) saved to {output_path}")
    return pose_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move an X5 arm and save TCP poses for hand-eye calibration."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--home-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="override home_joints_deg from the X5 config",
    )
    parser.add_argument("--speed-ratio", type=float, default=0.5)
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=50.0,
        help="maximum joint delta per HTTP move command (must not exceed server limit)",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="start the interactive prompt without moving home",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not (0.0 < args.speed_ratio <= 1.0):
        raise SystemExit("--speed-ratio must be in (0, 1]")
    if args.max_step_deg <= 0.0:
        raise SystemExit("--max-step-deg must be positive")

    home_joints_deg = (
        list(args.home_joints_deg)
        if args.home_joints_deg is not None
        else load_home_joints_deg(args.config, args.arm)
    )
    joint_limits_deg = load_joint_limits_deg(args.config)
    output_path = args.output or Path(
        f"RobotToolPose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    with X5HTTPClient(args.server_url, timeout_s=args.timeout) as client:
        try:
            run_interactive_session(
                client,
                arm=args.arm,
                home_joints_deg=home_joints_deg,
                output_path=output_path,
                speed_ratio=args.speed_ratio,
                max_step_deg=args.max_step_deg,
                joint_limits_deg=joint_limits_deg,
                skip_home=args.skip_home,
            )
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted. Sending stop command...")
            try:
                client.stop(args.arm)
            except Exception as stop_error:
                print(f"Warning: failed to stop {args.arm} arm: {stop_error}")
        except Exception:
            try:
                client.stop(args.arm)
            except Exception:
                pass
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 标定姿态
# -49.000 58.000 -18.000 90.000 0.000 100.000 -18.000
# -52.000 58.000 -18.000 90.000 0.000 100.000 -18.000
# -52.000 58.000 -23.000 90.000 0.000 100.000 -18.000
# -52.000 58.000 -23.000 85.000 0.000 100.000 -18.000
# -52.000 53.000 -23.000 85.000 5.000 100.000 -13.000
# -47.000 53.000 -23.000 85.000 5.000 100.000 -10.000
# -47.000 58.000 -23.000 85.000 5.000 100.000 -20.000
# -47.000 63.000 -28.000 95.000 5.000 100.000 -25.000
# -43.000 63.000 -28.000 95.000 5.000 100.000 -25.000
# -50.000 63.000 -28.000 95.000 5.000 95.000 -25.000
# -55.000 63.000 -28.000 95.000 10.000 95.000 -25.000
# -60.000 68.000 -32.000 95.000 10.000 95.000 -25.000
# -55.000 68.000 -32.000 95.000 10.000 90.000 -30.000
# -58.000 68.000 -32.000 100.000 15.000 90.000 -30.000
# -58.000 68.000 -32.000 100.000 15.000 85.000 -10.000