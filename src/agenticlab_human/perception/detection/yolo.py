"""Quick local smoke test for the fine-tuned closed-set YOLO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agenticlab_human.perception.detection.yolo_detector import YOLODETECTOR


DEFAULT_MODEL = (
    REPO_ROOT
    / "train_yolo/runs/agenticlab_objects_6cls_red_bin_yolo26s_ndjson/weights/best.pt"
)
DEFAULT_CLASSES = [
    "number_block_1",
    "number_block_2",
    "number_block_3",
    "paper_cup",
    "yellow_bin",
    "red_bin",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the new 6-class YOLO checkpoint on one image or a folder."
    )
    parser.add_argument(
        "--source",
        "-s",
        default=None,
        help=(
            "Image or directory to test. If omitted, the latest "
            "output/execution/*/pick/*_rgb.png capture is used."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Alias for --source when testing a single image.",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--limit", type=int, default=None, help="Max images from a directory.")
    parser.add_argument(
        "--output-dir",
        default="output/perception/yolo_test",
        help="Directory for detection_result.json, visualization.png, and original_img.png.",
    )
    return parser


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _latest_capture() -> Path:
    capture_root = REPO_ROOT / "output/execution"
    candidates = list(capture_root.glob("*/pick/*_rgb.png"))
    if not candidates:
        raise FileNotFoundError(
            "No --source was provided and no output/execution/*/pick/*_rgb.png "
            "capture exists."
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _iter_images(source: Path, limit: int | None = None) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")

    images = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit is not None:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No images found under: {source}")
    return images


def _safe_save_dir(base_dir: Path, image_path: Path, index: int) -> Path:
    name = image_path.stem.replace(" ", "_")
    candidate = base_dir / f"{index:03d}_{name}"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        retry = base_dir / f"{index:03d}_{name}_{suffix}"
        if not retry.exists():
            return retry
        suffix += 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_arg = args.image or args.source
    source = _repo_path(source_arg) if source_arg else _latest_capture()
    model_path = _repo_path(args.model_path)
    output_dir = _repo_path(args.output_dir)

    detector = YOLODETECTOR(
        model_path=str(model_path),
        confidence=args.conf,
        image_size=args.imgsz,
        output_dir=str(output_dir),
        session_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    image_paths = _iter_images(source, limit=args.limit)
    run_dir = Path(detector.session_output_dir)
    summaries = []
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        result = detector.detect(image, args.classes)
        save_dir = detector.save_detection(
            image,
            result,
            save_dir=str(_safe_save_dir(run_dir, image_path, index)),
        )
        summaries.append(
            {
                "image": str(image_path),
                "success": result.success,
                "num_objects": result.num_objects,
                "labels": result.get_all_labels,
                "save_dir": save_dir,
            }
        )

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "classes": args.classes,
                "confidence": args.conf,
                "image_size": args.imgsz,
                "num_images": len(image_paths),
                "run_dir": str(run_dir),
                "results": summaries,
            },
            indent=2,
        )
    )
    return 0 if any(item["success"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
