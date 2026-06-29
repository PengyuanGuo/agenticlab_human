import time
# ================================
# 语音活动检测模块
# ================================

class VoiceActivityDetector:
    """基础语音活动检测器 - 基于能量检测"""
    
    def __init__(self, 
                 silence_timeout=0.8,      # 静音超时时间（秒）- 更快结束
                 min_speech_duration=0.6,  # 最小语音持续时间（秒）- 更短语音
                 energy_threshold=0.5,   # 能量阈值 - 更敏感
                 calibration_duration=2.0, # 背景噪声校准时间（秒）
                 sample_rate=16000,        # 采样率
                 chunk_size=1024):         # 块大小
        
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.energy_threshold = energy_threshold
        self.calibration_duration = calibration_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # 状态变量
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.background_noise_level = 0.0
        self.calibrated = False
        
        # 历史能量数据（用于背景噪声估计）
        self.energy_history = []
        self.max_history_length = 100
        
        print(f"VAD初始化: 静音超时={silence_timeout}s, 最小语音={min_speech_duration}s, 能量阈值={energy_threshold}")
    
    def _calculate_energy(self, audio_data):
        """计算音频数据的RMS能量"""
        try:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            
            # 计算RMS能量
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return rms / 32768.0  # 归一化到0-1范围
        except Exception as e:
            print(f"计算能量失败: {e}")
            return 0.0
    
    def calibrate_background_noise(self, audio_data):
        """校准背景噪声水平"""
        energy = self._calculate_energy(audio_data)
        self.energy_history.append(energy)
        
        # 保持历史数据在合理范围内
        if len(self.energy_history) > self.max_history_length:
            self.energy_history.pop(0)
        
        if not self.calibrated and len(self.energy_history) >= 10:
            # 使用历史数据计算背景噪声水平
            import statistics
            self.background_noise_level = statistics.median(self.energy_history)
            
            # 动态调整阈值
            dynamic_threshold = self.background_noise_level * 2.5
            if dynamic_threshold > self.energy_threshold:
                self.energy_threshold = min(dynamic_threshold, 0.01)
                print(f"VAD动态调整阈值: {self.energy_threshold:.4f} (背景噪声: {self.background_noise_level:.4f})")
            
            self.calibrated = True
    
    def process_audio(self, audio_data):
        """处理音频数据，返回是否应该继续录音"""
        current_time = time.time()
        energy = self._calculate_energy(audio_data)
        
        # 如果还未校准，先进行背景噪声校准
        if not self.calibrated:
            self.calibrate_background_noise(audio_data)
            return True
        
        # 检测语音活动
        is_voice_active = energy > self.energy_threshold
        
        if is_voice_active:
            # 检测到语音
            if not self.is_speaking:
                # 语音开始
                self.is_speaking = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print("🎤 检测到语音开始")
            else:
                # 语音继续，重置静音计时
                self.silence_start_time = None
        else:
            # 静音状态
            if self.is_speaking:
                # 如果之前在说话，现在开始静音
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
    
                else:
                    # 检查静音持续时间
                    silence_duration = current_time - self.silence_start_time
                    

                    
                    if silence_duration >= self.silence_timeout:
                        # 静音超时，检查是否满足最小语音持续时间
                        if self.speech_start_time:
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= self.min_speech_duration:
                                print(f"🔇 检测到语音结束 (语音时长: {speech_duration:.1f}s, 静音时长: {silence_duration:.1f}s)")
                                return False
                            else:
                                print(f"⚠️ 语音时长过短 ({speech_duration:.1f}s < {self.min_speech_duration}s), 继续录音")
                                # 重置状态，继续录音
                                self.is_speaking = False
                                self.speech_start_time = None
                                self.silence_start_time = None
        
        return True
    
    def reset(self):
        """重置VAD状态"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        print("🔄 VAD状态已重置")


class SileroVAD:
    """Silero VAD封装类 - 支持重采样和音频累积"""
    
    def __init__(self, input_sample_rate=48000, target_sample_rate=16000):
        """初始化Silero VAD"""
        self.input_sample_rate = input_sample_rate    # 原始采样率
        self.target_sample_rate = target_sample_rate  # Silero VAD优化采样率
        self.model = None
        self.utils = None
        
        # 音频累积缓冲区 - 严格按照Silero VAD要求
        # Silero VAD严格要求：16000Hz -> 512样本, 8000Hz -> 256样本
        if target_sample_rate == 16000:
            self.target_samples = 512
        elif target_sample_rate == 8000:
            self.target_samples = 256
        else:
            self.target_samples = 512  # 默认
            
        self.audio_buffer = []
        print(f"🎯 Silero VAD配置: {target_sample_rate}Hz, {self.target_samples}样本")
        
        # if not deps.torch_available or not deps.silero_vad_available:
        #     raise ImportError("Silero VAD不可用，请安装torch: pip install torch")
        
        self._load_model()
    
    def _load_model(self):
        """加载Silero VAD模型 - 强制离线模式"""
        try:
            import torch
            import os
            print("🔄 正在加载Silero VAD模型（离线模式）...")
            
            # 设置环境变量强制离线模式
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            
            # 方法1: 直接使用本地缓存路径
            cache_dir = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
            if os.path.exists(cache_dir):
                print(f"📁 使用本地缓存: {cache_dir}")
                self.model, self.utils = torch.hub.load(
                    repo_or_dir=cache_dir,
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                    source='local'
                )
            else:
                # 方法2: 使用标准方式但设置离线模式
                print("🔄 尝试标准加载（离线模式）...")
                # 设置离线模式环境变量
                os.environ['TORCH_HUB_OFFLINE'] = '1'
                
                self.model, self.utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True
                )
            
            print("✅ Silero VAD模型加载成功")
                
        except Exception as e:
            print(f"❌ Silero VAD模型加载失败: {e}")
            print("💡 提示: 请确保已下载Silero VAD模型到本地缓存")
            print("   可以运行: python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')\"")
            raise
    
    def _resample_audio(self, audio_array):
        """重采样音频从input_sample_rate到target_sample_rate"""
        import numpy as np
        
        if self.input_sample_rate == self.target_sample_rate:
            return audio_array
        
        try:
            # 使用简单的线性插值重采样
            ratio = self.target_sample_rate / self.input_sample_rate
            new_length = int(len(audio_array) * ratio)
            
            if new_length == 0:
                return np.array([], dtype=audio_array.dtype)
            
            # 线性插值重采样
            old_indices = np.linspace(0, len(audio_array) - 1, new_length)
            new_audio = np.interp(old_indices, np.arange(len(audio_array)), audio_array)
            
            return new_audio.astype(audio_array.dtype)
            
        except Exception as e:
            print(f"⚠️ 音频重采样失败: {e}")
            return audio_array
    
    def detect_speech(self, audio_chunk):
        """检测音频块中的语音活动 - 支持累积和重采样"""
        try:
            import torch
            import numpy as np
            
            # 确保音频是numpy数组格式
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_array = audio_chunk.copy()
            
            # 添加到缓冲区
            self.audio_buffer.extend(audio_array)
            
            # 计算需要多少原始采样率的样本才能重采样到target_samples
            required_input_samples = int(self.target_samples * (self.input_sample_rate / self.target_sample_rate))
            
            # 检查是否有足够的数据进行检测
            if len(self.audio_buffer) < required_input_samples:
                return 0.0  # 数据不够，返回无语音
            
            # 精确取出所需的样本数
            window_audio = np.array(self.audio_buffer[:required_input_samples], dtype=np.int16)
            
            # 从缓冲区移除已处理的数据（减少重叠提高响应速度）
            overlap = required_input_samples // 4  # 从50%减少到25%重叠
            self.audio_buffer = self.audio_buffer[required_input_samples - overlap:]
            
            # 重采样到Silero VAD要求的精确样本数
            resampled_audio = self._resample_audio(window_audio)
            
            # 确保重采样后的样本数严格等于target_samples
            if len(resampled_audio) != self.target_samples:
                if len(resampled_audio) > self.target_samples:
                    resampled_audio = resampled_audio[:self.target_samples]
                else:
                    # 填充到目标长度
                    padded_array = np.zeros(self.target_samples, dtype=resampled_audio.dtype)
                    padded_array[:len(resampled_audio)] = resampled_audio
                    resampled_audio = padded_array
            
            # 转换为float32并归一化
            if resampled_audio.dtype != np.float32:
                resampled_audio = resampled_audio.astype(np.float32) / 32768.0
            
            # 验证样本数（调试用）
            if len(resampled_audio) != self.target_samples:
                print(f"⚠️ 样本数不匹配: 期望{self.target_samples}，实际{len(resampled_audio)}")
                return 0.0
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(resampled_audio)
            
            # 使用模型检测
            speech_prob = self.model(audio_tensor, self.target_sample_rate).item()
            
            return speech_prob
            
        except Exception as e:
            print(f"⚠️ Silero VAD检测失败: {e}")
            return 0.0
    
    def detect_speech_from_buffer(self, audio_data):
        """从预处理的音频数据直接检测语音 - 用于环形缓冲区"""
        try:
            import torch
            import numpy as np
            
            # 确保音频是numpy数组格式
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data.copy()
            
            # 重采样到Silero VAD要求的精确样本数
            resampled_audio = self._resample_audio(audio_array)
            
            # 确保重采样后的样本数严格等于target_samples
            if len(resampled_audio) != self.target_samples:
                if len(resampled_audio) > self.target_samples:
                    resampled_audio = resampled_audio[:self.target_samples]
                else:
                    # 填充到目标长度
                    padded_array = np.zeros(self.target_samples, dtype=resampled_audio.dtype)
                    padded_array[:len(resampled_audio)] = resampled_audio
                    resampled_audio = padded_array
            
            # 转换为float32并归一化
            if resampled_audio.dtype != np.float32:
                resampled_audio = resampled_audio.astype(np.float32) / 32768.0
            
            # 验证样本数（调试用）
            if len(resampled_audio) != self.target_samples:
                print(f"⚠️ 样本数不匹配: 期望{self.target_samples}，实际{len(resampled_audio)}")
                return 0.0
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(resampled_audio)
            
            # 使用模型检测
            speech_prob = self.model(audio_tensor, self.target_sample_rate).item()
            
            return speech_prob
            
        except Exception as e:
            print(f"⚠️ Silero VAD检测失败: {e}")
            return 0.0
    
    def reset_buffer(self):
        """重置音频缓冲区"""
        self.audio_buffer = []
    
    def is_speech(self, audio_chunk, threshold=0.5):
        """判断音频块是否包含语音"""
        prob = self.detect_speech(audio_chunk)
        return prob > threshold

    def deep_reset(self):
        """深度重置Silero VAD - 彻底清理所有内部状态"""
        try:
            import gc
            
            print("🔄 执行Silero VAD深度重置...")
            
            # 1. 清空音频缓冲区
            self.audio_buffer = []
            
            # 2. 强制垃圾回收
            gc.collect()
            
            # 3. 重新加载模型（如果可能）
            try:
                if self.model is not None:
                    # 尝试重新初始化模型
                    import torch
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("   ✅ Silero VAD模型状态已重置")
            except Exception as e:
                print(f"   ⚠️ 模型重置异常: {e}")
            
            # 4. 重新加载工具类（如果需要）
            try:
                if self.utils is None:
                    # 如果工具类丢失，尝试重新加载（离线模式）
                    import torch
                    import os
                    # 设置环境变量强制离线模式
                    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
                    
                    try:
                        # 尝试使用本地缓存
                        cache_dir = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
                        if os.path.exists(cache_dir):
                            _, self.utils = torch.hub.load(
                                repo_or_dir=cache_dir,
                                model='silero_vad',
                                force_reload=False,
                                onnx=False,
                                trust_repo=True,
                                source='local'
                            )
                        else:
                            # 使用标准方式但设置离线模式
                            os.environ['TORCH_HUB_OFFLINE'] = '1'
                            _, self.utils = torch.hub.load(
                                repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False,
                                onnx=False,
                                trust_repo=True
                            )
                        print("   ✅ Silero VAD工具类已重新加载")
                    except Exception as e:
                        print(f"   ⚠️ 工具类重新加载失败: {e}")
            except Exception as e:
                print(f"   ⚠️ 工具类重新加载异常: {e}")
            
            # 5. 验证重置结果
            if len(self.audio_buffer) == 0:
                print("   ✅ Silero VAD深度重置成功")
                return True
            else:
                print("   ❌ Silero VAD重置失败")
                return False
                
        except Exception as e:
            print(f"   ❌ Silero VAD深度重置异常: {e}")
        return False
    
    def health_check(self):
        """Silero VAD健康检测 - 检查内部状态是否正常"""
        try:
            health_status = {
                'model_loaded': self.model is not None,
                'utils_loaded': self.utils is not None,
                'buffer_empty': len(self.audio_buffer) == 0,
                'buffer_size': len(self.audio_buffer),
                'target_samples': self.target_samples,
                'input_sample_rate': self.input_sample_rate,
                'target_sample_rate': self.target_sample_rate
            }
            
            # 检查关键状态
            issues = []
            if not health_status['model_loaded']:
                issues.append("模型未加载")
            if not health_status['utils_loaded']:
                issues.append("工具类未加载")
            if health_status['buffer_size'] > 10000:  # 缓冲区过大
                issues.append(f"缓冲区过大: {health_status['buffer_size']}样本")
            
            if issues:
                print(f"⚠️ Silero VAD健康检查发现问题: {', '.join(issues)}")
                return False
            else:
                print(f"✅ Silero VAD健康检查通过: 缓冲区{health_status['buffer_size']}样本")
                return True
                
        except Exception as e:
            print(f"❌ Silero VAD健康检查异常: {e}")
            return False


class CircularBuffer:
    """环形缓冲区用于音频数据"""
    
    def __init__(self, max_duration_seconds, sample_rate, channels=1, dtype=None):
        """初始化环形缓冲区
        
        Args:
            max_duration_seconds: 最大缓存时长（秒）
            sample_rate: 采样率
            channels: 声道数
            dtype: 数据类型
        """
        import numpy as np
        
        if dtype is None:
            dtype = np.int16
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        # 计算缓冲区大小
        self.max_samples = int(max_duration_seconds * sample_rate * channels)
        self.buffer = np.zeros(self.max_samples, dtype=dtype)
        
        # 缓冲区指针
        self.write_pos = 0
        self.is_full = False
        
        print(f"🔄 环形缓冲区初始化: {max_duration_seconds}s, {self.max_samples}样本")
    
    def write(self, data):
        """写入音频数据"""
        import numpy as np
        
        if isinstance(data, bytes):
            audio_array = np.frombuffer(data, dtype=self.dtype)
        else:
            audio_array = data.astype(self.dtype)
        
        data_length = len(audio_array)
        
        if data_length == 0:
            return
        
        # 如果数据长度超过缓冲区大小，只保留最后的部分
        if data_length >= self.max_samples:
            self.buffer = audio_array[-self.max_samples:].copy()
            self.write_pos = 0
            self.is_full = True
            return
        
        # 计算写入位置
        end_pos = self.write_pos + data_length
        
        if end_pos <= self.max_samples:
            # 数据可以直接写入，不需要环绕
            self.buffer[self.write_pos:end_pos] = audio_array
        else:
            # 需要环绕写入
            first_part_size = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_array[:first_part_size]
            self.buffer[:data_length - first_part_size] = audio_array[first_part_size:]
            self.is_full = True
        
        self.write_pos = end_pos % self.max_samples
        if end_pos >= self.max_samples:
            self.is_full = True
    
    def read_last_seconds(self, duration_seconds):
        """读取最后N秒的音频数据"""
        import numpy as np
        
        samples_needed = int(duration_seconds * self.sample_rate * self.channels)
        samples_needed = min(samples_needed, self.max_samples)
        
        if not self.is_full and self.write_pos < samples_needed:
            # 缓冲区还未满，且数据不够
            return self.buffer[:self.write_pos].copy()
        
        if samples_needed >= self.max_samples:
            # 需要全部数据
            if self.is_full:
                # 按正确顺序重新排列数据
                result = np.zeros(self.max_samples, dtype=self.dtype)
                result[:self.max_samples - self.write_pos] = self.buffer[self.write_pos:]
                result[self.max_samples - self.write_pos:] = self.buffer[:self.write_pos]
                return result
            else:
                return self.buffer[:self.write_pos].copy()
        
        # 读取最后的samples_needed个样本
        if self.is_full:
            # 缓冲区已满，需要计算正确的读取位置
            start_pos = (self.write_pos - samples_needed) % self.max_samples
            
            if start_pos + samples_needed <= self.max_samples:
                # 数据连续，直接读取
                return self.buffer[start_pos:start_pos + samples_needed].copy()
            else:
                # 数据跨越边界，需要分两段读取
                result = np.zeros(samples_needed, dtype=self.dtype)
                first_part_size = self.max_samples - start_pos
                result[:first_part_size] = self.buffer[start_pos:]
                result[first_part_size:] = self.buffer[:samples_needed - first_part_size]
                return result
        else:
            # 缓冲区未满
            start_pos = max(0, self.write_pos - samples_needed)
            return self.buffer[start_pos:self.write_pos].copy()
    
    def clear(self):
        """清空缓冲区"""
        import numpy as np
        self.buffer.fill(0)
        self.write_pos = 0
        self.is_full = False
    
    def get_sample_count(self):
        """获取当前缓冲区中的样本数量"""
        if self.is_full:
            return self.max_samples
        else:
            return self.write_pos
    
    def read_last_samples(self, sample_count):
        """读取最后N个样本"""
        import numpy as np
        
        # 确保不超过缓冲区大小
        sample_count = min(sample_count, self.max_samples)
        
        if not self.is_full and self.write_pos < sample_count:
            # 缓冲区还未满，且数据不够
            return None
        
        # 计算读取位置
        if self.is_full:
            # 缓冲区已满，需要计算正确的读取位置
            start_pos = (self.write_pos - sample_count) % self.max_samples
            
            if start_pos + sample_count <= self.max_samples:
                # 数据连续，直接读取
                return self.buffer[start_pos:start_pos + sample_count].copy()
            else:
                # 数据跨越边界，需要分两段读取
                result = np.zeros(sample_count, dtype=self.dtype)
                first_part_size = self.max_samples - start_pos
                result[:first_part_size] = self.buffer[start_pos:]
                result[first_part_size:] = self.buffer[:sample_count - first_part_size]
                return result
        else:
            # 缓冲区未满
            start_pos = max(0, self.write_pos - sample_count)
            return self.buffer[start_pos:self.write_pos].copy()
    
    def get_last_samples_range(self, sample_count):
        """获取最后N个样本在缓冲区中的范围（用于重合率计算）"""
        # 确保不超过缓冲区大小
        sample_count = min(sample_count, self.max_samples)
        
        if not self.is_full and self.write_pos < sample_count:
            # 缓冲区还未满，且数据不够
            return None
        
        # 计算读取位置
        if self.is_full:
            # 缓冲区已满，需要计算正确的读取位置
            start_pos = (self.write_pos - sample_count) % self.max_samples
            end_pos = self.write_pos
        else:
            # 缓冲区未满
            start_pos = max(0, self.write_pos - sample_count)
            end_pos = self.write_pos
        
        return (start_pos, end_pos, sample_count)


class AdvancedVoiceActivityDetector:
    """高级语音活动检测器 - 基于Silero VAD + 环形缓冲区"""
    
    def __init__(self, 
                 sample_rate=16000,
                 buffer_duration=3.0,      # 环形缓冲区时长（秒）
                 lookback_duration=0.5,    # 回溯时长（秒）
                 silence_timeout=1.0,      # 静音超时（秒）
                 min_speech_duration=0.4,  # 最小语音时长（秒）- 允许更短语音
                 speech_threshold=0.3,     # Silero VAD阈值 - 提高敏感度
                 overlap_threshold=0.8):   # 样本重合率阈值 - 避免重复检测
        
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.lookback_duration = lookback_duration
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.speech_threshold = speech_threshold
        self.overlap_threshold = overlap_threshold
        
        # 初始化组件 - 启用优化版Silero VAD
        # if deps.silero_vad_available and deps.torch_available:
        try:
            # 使用音频累积+重采样的Silero VAD
            self.silero_vad = SileroVAD(
                input_sample_rate=sample_rate,    # 你的实际采样率（48000Hz）
                target_sample_rate=16000          # Silero VAD优化采样率
            )
            self.vad_available = True
            print("✅ 使用优化版Silero VAD（音频累积+重采样）")
            print(f"📊 采样率转换: {sample_rate}Hz → 16000Hz")
        except Exception as e:
            print(f"⚠️ Silero VAD初始化失败: {e}")
            print("⚠️ 回退到基础能量VAD")
            self.silero_vad = None
            self.vad_available = False
        
        # 初始化环形缓冲区
        import numpy as np
        self.buffer = CircularBuffer(
            max_duration_seconds=buffer_duration,
            sample_rate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        
        # 状态管理
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_detection_samples = None  # 记录上次检测的样本范围
        
        # 回退VAD（能量检测）
        if not self.vad_available:
            self.fallback_vad = VoiceActivityDetector(
                silence_timeout=silence_timeout,
                min_speech_duration=min_speech_duration,
                sample_rate=sample_rate
            )
        
        print(f"🎯 高级VAD初始化完成:")
        print(f"   缓冲区: {buffer_duration}s, 回溯: {lookback_duration}s")
        print(f"   静音超时: {silence_timeout}s, 最小语音: {min_speech_duration}s")
        print(f"   重合率阈值: {overlap_threshold}")
    
    def feed_audio(self, audio_data):
        """向环形缓冲区输入音频数据，并基于数据累积触发VAD检测"""
        self.buffer.write(audio_data)
        
        # 检查环形缓冲区是否有足够数据进行VAD检测
        if self.vad_available:
            return self._check_voice_activity_with_silero()
        else:
            return self._check_voice_activity_with_fallback(audio_data)
    
    def _check_voice_activity_with_silero(self):
        """使用Silero VAD检查语音活动状态 - 基于环形缓冲区数据"""
        current_time = time.time()
        
        # 检查Silero VAD是否可用
        if not self.silero_vad or not hasattr(self.silero_vad, 'target_samples'):
            print("⚠️ Silero VAD不可用，回退到基础VAD")
            return None
        
        # 计算Silero VAD需要的输入样本数
        required_input_samples = int(self.silero_vad.target_samples * 
                                   (self.sample_rate / self.silero_vad.target_sample_rate))
        
        # 检查环形缓冲区是否有足够数据
        if self.buffer.get_sample_count() < required_input_samples:
            return None  # 数据不足，等待更多数据
        
        # 检查样本重合率，避免重复检测
        current_range = self.buffer.get_last_samples_range(required_input_samples)
        if current_range is None:
            return None  # 数据不足
        
        if self._is_overlap_too_high(current_range):
            return None  # 重合率过高，跳过本次检测
        
        # 从环形缓冲区取出最新的足够样本进行检测
        window_audio = self.buffer.read_last_samples(required_input_samples)
        
        if window_audio is None or len(window_audio) < required_input_samples:
            return None  # 数据不足
        
        # 使用Silero VAD检测
        try:
            speech_prob = self.silero_vad.detect_speech_from_buffer(window_audio)
            is_speech = speech_prob > self.speech_threshold
        except Exception as e:
            print(f"⚠️ Silero VAD检测失败: {e}")
            return None
        
        # 更新上次检测的样本范围
        self.last_detection_samples = current_range
        
        return self._process_vad_result(is_speech, current_time)
    
    def _is_overlap_too_high(self, current_range):
        """检查当前检测样本与上次检测样本的重合率是否过高"""
        if self.last_detection_samples is None:
            return False  # 第一次检测，没有重合
        
        # 解包范围信息
        current_start, current_end, current_count = current_range
        last_start, last_end, last_count = self.last_detection_samples
        
        # 计算重合的样本数
        overlap_count = 0
        
        # 处理环形缓冲区的边界情况
        if self.buffer.is_full:
            # 缓冲区已满，需要考虑环形边界
            if current_start <= current_end:
                # 当前范围不跨越边界
                current_samples = set(range(current_start, current_end))
            else:
                # 当前范围跨越边界
                current_samples = set(range(current_start, self.buffer.max_samples)) | set(range(0, current_end))
            
            if last_start <= last_end:
                # 上次范围不跨越边界
                last_samples = set(range(last_start, last_end))
            else:
                # 上次范围跨越边界
                last_samples = set(range(last_start, self.buffer.max_samples)) | set(range(0, last_end))
        else:
            # 缓冲区未满，简单范围比较
            current_samples = set(range(current_start, current_end))
            last_samples = set(range(last_start, last_end))
        
        # 计算重合样本数
        overlap_count = len(current_samples & last_samples)
        
        # 计算重合率
        overlap_ratio = overlap_count / min(current_count, last_count)
        
        # 调试信息
        if overlap_ratio > 0.5:  # 只在重合率较高时打印调试信息
            print(f"🔍 样本重合率: {overlap_ratio:.2f} ({overlap_count}/{min(current_count, last_count)})")
        
        return overlap_ratio > self.overlap_threshold
    
    def _check_voice_activity_with_fallback(self, latest_audio):
        """使用回退VAD检查语音活动状态"""
        current_time = time.time()
        
        # 转换为字节格式供回退VAD使用
        if isinstance(latest_audio, bytes):
            audio_bytes = latest_audio
        else:
            import numpy as np
            if latest_audio.dtype != np.int16:
                audio_int16 = (latest_audio * 32767).astype(np.int16)
            else:
                audio_int16 = latest_audio
            audio_bytes = audio_int16.tobytes()
        
        # 使用回退VAD的能量检测
        energy = self.fallback_vad._calculate_energy(audio_bytes)
        is_speech = energy > 0.005  # 使用较低的阈值
        
        return self._process_vad_result(is_speech, current_time)
    
    def _process_vad_result(self, is_speech, current_time):
        """处理VAD检测结果的状态机逻辑"""
        if is_speech:
            if not self.is_speech_active:
                # 语音开始 - 立即触发
                self.is_speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"🎤 检测到语音开始 ({'Silero' if self.vad_available else 'Energy'} VAD)")
                return "speech_start"
            else:
                # 语音继续
                self.silence_start_time = None
                return "speech_continue"
        else:
            if self.is_speech_active:
                # 可能的语音结束
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    return "silence_start"
                else:
                    # 检查静音持续时间
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.silence_timeout:
                        # 语音结束
                        if self.speech_start_time:
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= self.min_speech_duration:
                                print(f"🔇 检测到语音结束 (语音: {speech_duration:.1f}s, 静音: {silence_duration:.1f}s)")
                                return "speech_end"
                            else:
                                print(f"⚠️ 语音过短 ({speech_duration:.1f}s), 忽略")
                                self._reset_state()
                                return "speech_too_short"
                        else:
                            self._reset_state()
                            return "speech_too_short"
                    else:
                        return "silence_continue"
        
        return "no_change"
    
    def get_speech_audio(self):
        """获取检测到的语音音频（包含回溯）"""
        if not self.is_speech_active or not self.speech_start_time:
            return None
    
        # 计算需要回溯的总时长
        current_time = time.time()
        speech_duration = current_time - self.speech_start_time
        
        # 增加额外的安全回溯时间，确保不丢失语音开头
        safe_lookback = self.lookback_duration + 0.3  # 额外300ms安全边界
        total_duration = speech_duration + safe_lookback
        
        # 从环形缓冲区读取音频
        audio_data = self.buffer.read_last_seconds(total_duration)
        
        if len(audio_data) == 0:
            return None
        
        print(f"📀 提取语音音频: {len(audio_data)}样本 ({len(audio_data)/self.sample_rate:.2f}s)")
        print(f"   📊 回溯详情: 语音时长{speech_duration:.2f}s + 安全回溯{safe_lookback:.2f}s = 总计{total_duration:.2f}s")
        return audio_data
    
    def _reset_state(self):
        """重置状态"""
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
    
    def reset(self):
        """重置检测器 - 增强版，包含Silero VAD深度重置"""
        self._reset_state()
        self.buffer.clear()
        
        # 深度重置Silero VAD（解决长时间运行后的状态问题）
        if self.vad_available and self.silero_vad:
            try:
                # 先尝试深度重置
                deep_reset_success = self.silero_vad.deep_reset()
                if deep_reset_success:
                    print("🔄 Silero VAD深度重置成功")
                else:
                    # 如果深度重置失败，尝试普通重置
                    print("⚠️ Silero VAD深度重置失败，尝试普通重置")
                    self.silero_vad.reset_buffer()
            except Exception as e:
                print(f"⚠️ Silero VAD重置异常: {e}")
                # 最后的回退方案
                try:
                    self.silero_vad.reset_buffer()
                except:
                    pass
        
        print("�� 高级VAD状态已重置（增强版）")