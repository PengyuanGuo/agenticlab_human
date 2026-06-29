import os
import tempfile
import time

from agenticlab_human.voice.backend.ASR import SenseVoiceRecognizer
from agenticlab_human.voice.backend.utils import SimplifiedVoiceRecorder
from agenticlab_human.voice.config.speech_config import SpeechConfig

class SpeechInputService:
    def __init__(self, device: str = SpeechConfig.DEVICE):
        print("🚀 正在初始化语音输入服务...")
        
        # 初始化录音器
        self.recorder = SimplifiedVoiceRecorder()
        print("🎤 录音器(SimplifiedVoiceRecorder) 初始化完成")
        
        # 预设临时文件集合，防止泄露
        self.temp_audio_files = set()

        # 确保保存目录存在
        if not os.path.exists(SpeechConfig.TEMP_AUDIO_DIR):
            os.makedirs(SpeechConfig.TEMP_AUDIO_DIR, exist_ok=True)
        
        # 初始化 ASR
        self.recognizer = SenseVoiceRecognizer(device=device)
        print("✅ SenseVoice语音识别引擎已启用")

    def listen(self) -> str:
        """
        开始监听语音并转换为文本。
        流程:
        1. 启动录音机 (含内部 VAD 检测机制)
        2. 检测到声音结束并保存 wav 录音文件
        3. 调用 ASR 将其转化为文本
        4. 自动清理生成的wav
        """
        try:
            print("\n👂 正在等待语音输入 (监听中)...")
            self.recorder.start_recording(use_vad=SpeechConfig.USE_VAD)
            
            # 保存到临时目录
            suffix = ".wav"
            tmp_file = tempfile.NamedTemporaryFile(dir=SpeechConfig.TEMP_AUDIO_DIR, suffix=suffix, delete=False)
            audio_filename = tmp_file.name
            tmp_file.close()

            self.temp_audio_files.add(audio_filename)
            
            if not self.recorder.save_audio(audio_filename):
                print("❌ 录音未成功保存或无有效声音")
                self.temp_audio_files.discard(audio_filename)
                if os.path.exists(audio_filename):
                    os.unlink(audio_filename)
                return ""
            
            # 使用 ASR 识别文本
            asr_start_time = time.time()
            try:
                text = self.recognizer.recognize(audio_filename)
                asr_end_time = time.time()
                print(f"⏱️ ASR语音转文字耗时: {asr_end_time - asr_start_time:.2f}秒")
                return text.strip()
            except Exception as e:
                print(f"⚠️ ASR语音识别异常: {e}")
                return ""
                
        except Exception as e:
            print(f"❌ 语音监听异常: {e}")
            return ""
        finally:
            # 清理刚才录制生成的临时文件
            if 'audio_filename' in locals():
                try:
                    if os.path.exists(audio_filename):
                        os.unlink(audio_filename)
                        self.temp_audio_files.discard(audio_filename)
                except Exception as del_e:
                    print(f"⚠️ 清理失效音频文件失败: {del_e}")
            
            if self.recorder:
                self.recorder.cleanup()
                
    def shutdown(self):
        """完全关闭服务并清理剩余文件"""
        print("🛑 正在关闭语音输入服务...")
        for file in list(self.temp_audio_files):
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except:
                pass
        self.temp_audio_files.clear()
