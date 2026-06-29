# 语音转文字模块配置

from pathlib import Path

class SpeechConfig:
    # 默认设备：cpu / cuda
    DEVICE = "cpu"
    
    # 临时文件保存路径
    TEMP_AUDIO_DIR = str(Path(__file__).resolve().parents[1] / "temp_audio")

    # VAD & 麦克风配置
    USE_VAD = True
