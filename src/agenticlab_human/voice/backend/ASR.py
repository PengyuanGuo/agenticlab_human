# ================================
# 语音识别模块
# ================================

import os
from pathlib import Path

class SenseVoiceRecognizer:
    """SenseVoice语音识别器"""
    
    def __init__(self, device="cpu"):
        # if not deps.funasr_available:
        #     raise ImportError("请先安装FunASR: pip install funasr")
            
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            # vad_model="fsmn-vad",
            # vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True
        )
        self.rich_transcription_postprocess = rich_transcription_postprocess
    
    def recognize(self, audio_file: str) -> str:
        """识别音频文件并返回文本"""
        try:
            res = self.model.generate(
                input=audio_file,
                cache={},
                language="zh",  # 中文
                use_itn=True,   # 使用逆文本标准化
                batch_size_s=60,
                # merge_vad=True,
                # merge_length_s=15,
            )
            text = self.rich_transcription_postprocess(res[0]["text"])
            return text
        except Exception as e:
            print(f"SenseVoice语音识别错误: {e}")
            return ""