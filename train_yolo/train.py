"""Train an AgenticLab YOLO detector.

Examples:
    python train_yolo/train.py \
      --data train_yolo/raw/frames/smoke_femtobolt/rgb/agenticlabtabletop1.ndjson

    python train_yolo/train.py \
      --data train_yolo/datasets/agenticlab_objects/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO on AgenticLab tabletop objects.")
    parser.add_argument(
        "--data",
        default="train_yolo/datasets/agenticlab_objects/data.yaml",
        help="Dataset config path. Can be Ultralytics data.yaml or .ndjson.",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", default=16, help="Batch size, or -1 for auto batch.")
    parser.add_argument("--device", default=0, help="Device passed to Ultralytics, e.g. 0, cpu, 0,1.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="train_yolo/runs/train")
    parser.add_argument("--name", default="agenticlab_objects_yolo26s")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    parser.add_argument("--no-val", action="store_true", help="Skip explicit validation after training.")
    return parser


def _parse_batch(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=_parse_batch(str(args.batch)),
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        cache=args.cache,
        single_cls=False,
    )

    if not args.no_val:
        model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
