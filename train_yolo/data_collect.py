"""Femto Bolt data collection helpers for AgenticLab YOLO training.

The intended workflow mirrors train_yolo/train.md:

    camera/video -> sparse frame extraction -> dedup -> pre-label -> human fix

This script keeps collection lightweight and repo-local. It saves RGB frames for
YOLO annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


AGENTICLAB_CLASSES = (
    "number_block_1",
    "number_block_2",
    "number_block_3",
    "paper_cup",
    "yellow_bin",
)
SUPPORTED_CAMERAS = ("Orbbec", "FemtoBolt", "Gemini305")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class SessionConfig:
    session: str
    camera: str
    fps: float
    start_delay_s: float
    scene: str
    lighting: str
    objects: list[str]
    notes: str
    output_root: str
    started_at: str


@dataclass
class FrameRecord:
    session: str
    frame_index: int
    timestamp: str
    rgb_path: str
    width: int
    height: int
    camera: str
    scene: str
    lighting: str
    objects: list[str]
    notes: str


def _timestamp_for_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _timestamp_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _resolve_output_root(path: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_rgb_jpg(rgb_image, path: Path, quality: int) -> None:
    Image = _require_pil_image()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_image).save(path, quality=int(quality))


def _require_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("This command requires Pillow. Install pillow or use the project environment.") from exc
    return Image


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("This command requires OpenCV. Install opencv-python or use the project environment.") from exc
    return cv2


def _load_camera_capture():
    try:
        from agenticlab_human.perception.camera.cam_capture import CameraCapture
    except ImportError as exc:
        raise ImportError(
            "Camera collection requires pyorbbecsdk and OpenCV. "
            "Use extract-video/dedup without the camera SDK, or run in the Femto Bolt environment."
        ) from exc
    return CameraCapture


def _show_preview(rgb_image) -> bool:
    cv2 = _require_cv2()
    display_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    canvas = cv2.resize(display_rgb, (640, 360))
    title = "Femto Bolt Data Collection: RGB - press q to stop"

    cv2.imshow(title, canvas)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def collect_from_camera(args: argparse.Namespace) -> int:
    cv2 = _require_cv2()
    CameraCapture = _load_camera_capture()

    session = args.session or f"{_timestamp_for_name()}_femtobolt"
    output_root = _resolve_output_root(args.output_root)
    rgb_dir = output_root / "frames" / session / "rgb"
    metadata_path = output_root / "frames" / session / "metadata.jsonl"
    session_path = output_root / "frames" / session / "session.json"

    objects = list(args.objects or AGENTICLAB_CLASSES)
    session_config = SessionConfig(
        session=session,
        camera=args.which_cam,
        fps=float(args.fps),
        start_delay_s=float(args.start_delay_s),
        scene=args.scene,
        lighting=args.lighting,
        objects=objects,
        notes=args.notes,
        output_root=_relative_to_repo(output_root),
        started_at=_timestamp_iso(),
    )
    _write_json(session_path, asdict(session_config))

    interval_s = 1.0 / float(args.fps)
    frame_index = 0
    saved_count = 0
    start_time = time.monotonic()
    next_save_time = start_time

    print(f"Starting {args.which_cam} collection session: {session}")
    print(f"RGB frames: {_relative_to_repo(rgb_dir)}")
    print("Press Ctrl+C to stop. With --preview, press q to stop.")

    try:
        with CameraCapture(args.which_cam) as camera:
            for _ in range(max(0, int(args.warmup_frames))):
                camera.capture()

            if args.start_delay_s > 0:
                print(f"Waiting {args.start_delay_s:g}s before saving frames. Arrange the scene now.")
                time.sleep(float(args.start_delay_s))
                start_time = time.monotonic()
                next_save_time = start_time

            while True:
                now = time.monotonic()
                if args.duration_s is not None and now - start_time >= args.duration_s:
                    break
                if args.max_frames is not None and saved_count >= args.max_frames:
                    break

                color_rgb, _ = camera.capture()
                if args.preview and _show_preview(color_rgb):
                    break

                if now < next_save_time:
                    continue
                next_save_time = now + interval_s

                stem = f"{frame_index:06d}"
                rgb_path = rgb_dir / f"{stem}.jpg"

                _save_rgb_jpg(color_rgb, rgb_path, args.jpg_quality)

                height, width = color_rgb.shape[:2]
                record = FrameRecord(
                    session=session,
                    frame_index=frame_index,
                    timestamp=_timestamp_iso(),
                    rgb_path=_relative_to_repo(rgb_path),
                    width=int(width),
                    height=int(height),
                    camera=args.which_cam,
                    scene=args.scene,
                    lighting=args.lighting,
                    objects=objects,
                    notes=args.notes,
                )
                _append_jsonl(metadata_path, asdict(record))

                saved_count += 1
                frame_index += 1
                print(f"saved {saved_count:04d}: {_relative_to_repo(rgb_path)}")
    except KeyboardInterrupt:
        print("\nCollection stopped by user.")
    finally:
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Done. Saved {saved_count} RGB frames.")
    return 0


def extract_video_frames(args: argparse.Namespace) -> int:
    cv2 = _require_cv2()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    session = args.session or f"{video_path.stem}_{_timestamp_for_name()}"
    output_root = _resolve_output_root(args.output_root)
    rgb_dir = output_root / "frames" / session / "rgb"
    metadata_path = output_root / "frames" / session / "metadata.jsonl"
    session_path = output_root / "frames" / session / "session.json"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(source_fps / float(args.fps))))
    objects = list(args.objects or AGENTICLAB_CLASSES)

    session_config = {
        "session": session,
        "source_video": _relative_to_repo(video_path),
        "source_fps": source_fps,
        "target_fps": float(args.fps),
        "frame_step": step,
        "scene": args.scene,
        "lighting": args.lighting,
        "objects": objects,
        "notes": args.notes,
        "started_at": _timestamp_iso(),
    }
    _write_json(session_path, session_config)

    saved_count = 0
    source_index = 0
    print(f"Extracting {args.fps:g} FPS from {_relative_to_repo(video_path)}")
    print(f"Output: {_relative_to_repo(rgb_dir)}")

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_index % step != 0:
                source_index += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stem = f"{saved_count:06d}"
            rgb_path = rgb_dir / f"{stem}.jpg"
            _save_rgb_jpg(rgb, rgb_path, args.jpg_quality)

            height, width = rgb.shape[:2]
            record = {
                "session": session,
                "frame_index": saved_count,
                "source_frame_index": source_index,
                "timestamp": _timestamp_iso(),
                "rgb_path": _relative_to_repo(rgb_path),
                "width": int(width),
                "height": int(height),
                "source_video": _relative_to_repo(video_path),
                "camera": args.which_cam,
                "scene": args.scene,
                "lighting": args.lighting,
                "objects": objects,
                "notes": args.notes,
            }
            _append_jsonl(metadata_path, record)

            saved_count += 1
            source_index += 1

            if args.max_frames is not None and saved_count >= args.max_frames:
                break
        print(f"Done. Saved {saved_count} frames from {total_frames or source_index} source frames.")
    finally:
        capture.release()
    return 0


def _average_hash(image_path: Path, hash_size: int = 8) -> tuple[bool, ...]:
    Image = _require_pil_image()
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    image = Image.open(image_path).convert("L").resize((hash_size, hash_size), resampling)
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    return tuple(pixel > mean for pixel in pixels)


def _hamming_distance(hash_a: tuple[bool, ...], hash_b: tuple[bool, ...]) -> int:
    return sum(a != b for a, b in zip(hash_a, hash_b))


def _iter_images(path: Path) -> Iterable[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in suffixes:
            yield image_path


def dedup_frames(args: argparse.Namespace) -> int:
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = REPO_ROOT / source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_hashes: list[tuple[bool, ...]] = []
    kept_count = 0
    skipped_count = 0

    print(f"Deduplicating {_relative_to_repo(source_dir)}")
    print(f"Output: {_relative_to_repo(output_dir)}")

    for image_path in _iter_images(source_dir):
        image_hash = _average_hash(image_path, hash_size=args.hash_size)
        is_duplicate = any(_hamming_distance(image_hash, kept_hash) <= args.threshold for kept_hash in kept_hashes)
        if is_duplicate:
            skipped_count += 1
            continue

        kept_hashes.append(image_hash)
        output_path = output_dir / f"{kept_count:06d}{image_path.suffix.lower()}"
        Image = _require_pil_image()
        with Image.open(image_path) as image:
            image.save(output_path)
        kept_count += 1

    print(f"Done. Kept {kept_count}, skipped {skipped_count}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Femto Bolt data collection pipeline for AgenticLab YOLO datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect sparse RGB frames from a Femto Bolt/Orbbec camera.")
    collect.add_argument("--which-cam", default="FemtoBolt", choices=SUPPORTED_CAMERAS)
    collect.add_argument("--session", default=None, help="Session name. Defaults to timestamp_femtobolt.")
    collect.add_argument("--output-root", default="train_yolo/raw")
    collect.add_argument("--fps", type=float, default=2.0, help="Saved frame rate for sparse collection.")
    collect.add_argument("--duration-s", type=float, default=None, help="Stop after this many seconds.")
    collect.add_argument("--max-frames", type=int, default=None, help="Stop after saving this many frames.")
    collect.add_argument("--warmup-frames", type=int, default=10)
    collect.add_argument("--start-delay-s", type=float, default=0.0, help="Wait before saving the first frame.")
    collect.add_argument("--preview", action="store_true", help="Show RGB preview. Press q to stop.")
    collect.add_argument("--jpg-quality", type=int, default=95)
    collect.add_argument("--scene", default="agenticlab_tabletop")
    collect.add_argument("--lighting", default="lab_normal")
    collect.add_argument("--objects", nargs="*", default=list(AGENTICLAB_CLASSES))
    collect.add_argument("--notes", default="")
    collect.set_defaults(func=collect_from_camera)

    extract = subparsers.add_parser("extract-video", help="Extract sparse RGB frames from a recorded video.")
    extract.add_argument("video", help="Input video path.")
    extract.add_argument("--which-cam", default="FemtoBolt", choices=SUPPORTED_CAMERAS)
    extract.add_argument("--session", default=None)
    extract.add_argument("--output-root", default="train_yolo/raw")
    extract.add_argument("--fps", type=float, default=2.0)
    extract.add_argument("--max-frames", type=int, default=None)
    extract.add_argument("--jpg-quality", type=int, default=95)
    extract.add_argument("--scene", default="agenticlab_tabletop")
    extract.add_argument("--lighting", default="lab_normal")
    extract.add_argument("--objects", nargs="*", default=list(AGENTICLAB_CLASSES))
    extract.add_argument("--notes", default="")
    extract.set_defaults(func=extract_video_frames)

    dedup = subparsers.add_parser("dedup", help="Copy perceptually unique frames into a new directory.")
    dedup.add_argument("source", help="Directory containing extracted RGB frames.")
    dedup.add_argument("output", help="Directory for deduplicated frames.")
    dedup.add_argument("--hash-size", type=int, default=8)
    dedup.add_argument("--threshold", type=int, default=4, help="Max Hamming distance considered duplicate.")
    dedup.set_defaults(func=dedup_frames)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "fps") and args.fps <= 0:
        raise ValueError("--fps must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
