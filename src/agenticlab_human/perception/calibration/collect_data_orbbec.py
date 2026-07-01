# coding=utf-8
import logging, os
import sys
import json
from datetime import datetime

import cv2
import numpy as np

from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline

_ORBBEC_EXAMPLES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "pyorbbecsdk", "examples")
)
if _ORBBEC_EXAMPLES not in sys.path:
    sys.path.insert(0, _ORBBEC_EXAMPLES)
from utils import frame_to_bgr_image

from libs.log_setting import CommonLog
from libs.auxiliary import popup_message

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


def _distortion_coeffs(dist) -> list:
    """Best-effort OpenCV-ordered coefficients from OBDistortion (k1,k2,p1,p2,k3,...)."""
    if dist is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    coeffs = [float(getattr(dist, n)) for n in names if hasattr(dist, n)]
    return coeffs if coeffs else [0.0, 0.0, 0.0, 0.0, 0.0]


def _serialize_color_intrinsics(color_frame) -> dict | None:
    """Match collect_data_realsense intrinsics.json layout (ppx/ppy, coeffs)."""
    video = color_frame.as_video_frame()
    if video is None:
        return None
    profile = video.get_stream_profile()
    vsp = profile.as_video_stream_profile()
    intr = vsp.get_intrinsic()
    print(intr)
    dist = vsp.get_distortion()
    ppx = float(getattr(intr, "cx", getattr(intr, "ppx", 0.0)))
    ppy = float(getattr(intr, "cy", getattr(intr, "ppy", 0.0)))
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": ppx,
        "ppy": ppy,
        "model": "orbbec_pinhole",
        "coeffs": _distortion_coeffs(dist),
    }


def displayFemtoBolt():
    """Femto Bolt (pyorbbecsdk): color capture for hand–eye data collection."""
    pipeline = Pipeline()
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            raise RuntimeError("未找到彩色传感器配置")
        try:
            color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
        except OBError as e:
            logger_.warning(f"1280x720 RGB@30 不可用，使用默认彩色配置: {e}")
            color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        logger_.error(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")
        sys.exit(1)

    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error(f"Pipeline 启动失败：{e}")
        popup_message("提醒", "相机连接异常")
        sys.exit(1)

    img_count = 0
    save_path = None
    last_color_frame = None

    base_save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(base_save_dir, exist_ok=True)

    logger_.info("相机已启动。按 'c' 键拍照，按 'ESC' 键退出，按 'i' 保存内参。")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            last_color_frame = color_frame
            original_image = frame_to_bgr_image(color_frame)
            if original_image is None:
                continue

            display_image = original_image.copy()
            cv2.putText(
                display_image,
                f"Captured: {img_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture_Video", display_image)

            k = cv2.waitKey(30) & 0xFF

            if k == 27:
                logger_.info(f"退出拍摄程序。总共拍摄了 {img_count} 张照片。")
                break

            elif k == ord("c"):
                if save_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(base_save_dir, timestamp)
                    os.makedirs(save_path)
                    logger_.info(f"创建新文件夹用于保存照片: {save_path}")

                image_name = f"img_{img_count:02d}.jpg"
                full_image_path = os.path.join(save_path, image_name)
                cv2.imwrite(full_image_path, original_image)
                logger_.info(f"照片已保存: {full_image_path}")
                img_count += 1

            elif k == ord("i"):
                if save_path is None:
                    logger_.warning("请先按 'c' 键拍摄至少一张照片以创建文件夹。")
                    continue
                if last_color_frame is None:
                    logger_.warning("尚未收到彩色帧，无法读取内参。")
                    continue

                intrinsics_dict = _serialize_color_intrinsics(last_color_frame)
                if intrinsics_dict is None:
                    logger_.error("无法从当前帧解析相机内参。")
                    continue

                json_path = os.path.join(save_path, "intrinsics.json")
                with open(json_path, "w") as f:
                    json.dump(intrinsics_dict, f, indent=4)

                logger_.info(f"相机内参已保存到: {json_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    displayFemtoBolt()
