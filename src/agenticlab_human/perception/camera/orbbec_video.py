"""
Simple Orbbec (Femto Bolt) color video recorder.

Usage:
  python -m agenticlab_human.perception.camera.orbbect_video --out output.mp4

Controls:
  - Press 'q' or ESC to stop recording
"""

from __future__ import annotations

import argparse
import os
import time

ESC_KEY = 27


def _make_video_writer(path: str, fps: int, size: tuple[int, int]):
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    ext = os.path.splitext(path)[1].lower()
    # Prefer a sane default for common containers.
    if ext in {".mp4", ".m4v"}:
        fourcc_candidates = ["mp4v", "avc1"]
    else:
        fourcc_candidates = ["XVID", "MJPG"]

    last_err: str | None = None
    for fourcc_str in fourcc_candidates:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), float(fps), size)
        if writer.isOpened():
            return writer
        last_err = f"failed to open VideoWriter with fourcc={fourcc_str}"

    raise RuntimeError(last_err or "failed to open VideoWriter")


def record_color_video(
    out_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    preview: bool = True,
    max_seconds: float | None = None,
) -> None:
    import cv2

    from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline

    from agenticlab_human.perception.camera.orbbec_utils import frame_to_bgr_image

    config = Config()
    pipeline = Pipeline()

    # Configure COLOR stream at requested resolution/fps.
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if profile_list is None:
        raise RuntimeError("No COLOR sensor profiles available")
    try:
        color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, int(fps))
    except OBError:
        color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    writer = _make_video_writer(out_path, fps=int(fps), size=(int(width), int(height)))

    if preview:
        cv2.namedWindow("Orbbec Recorder", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Orbbec Recorder", width, height)

    start_t = time.time()
    frames_written = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            bgr = frame_to_bgr_image(color_frame)
            if bgr is None:
                continue

            # Make sure writer gets exactly WxH frames.
            if bgr.shape[1] != width or bgr.shape[0] != height:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            writer.write(bgr)
            frames_written += 1

            if preview:
                cv2.imshow("Orbbec Recorder", bgr)
                key = cv2.waitKey(1)
                if key in (ord("q"), ESC_KEY):
                    break

            if max_seconds is not None and (time.time() - start_t) >= float(max_seconds):
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            writer.release()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        if preview:
            cv2.destroyAllWindows()

    duration = max(1e-6, time.time() - start_t)
    print(
        f"Saved: {out_path}\n"
        f"Frames: {frames_written}\n"
        f"Approx FPS: {frames_written / duration:.2f}\n"
        f"Resolution: {width}x{height}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Femto Bolt color video (Orbbec).")
    parser.add_argument("--out", required=True, help="Output video path, e.g. output.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window.")
    parser.add_argument("--seconds", type=float, default=None, help="Stop after N seconds.")
    args = parser.parse_args()

    record_color_video(
        out_path=args.out,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=not args.no_preview,
        max_seconds=args.seconds,
    )


if __name__ == "__main__":
    main()

