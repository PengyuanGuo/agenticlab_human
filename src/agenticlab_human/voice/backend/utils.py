import time
from .VAD import AdvancedVoiceActivityDetector, VoiceActivityDetector
import os
import threading
# ================================
# 跨平台音频处理模块
# ================================

class CrossPlatformAudioManager:
    """跨平台音频管理器 - 使用sounddevice和soundfile"""
    
    def __init__(self):
        self.channels = 1
        self.dtype = 'int16'
        
        # 预选择最佳设备
        self.input_device = self._get_best_input_device()
        self.output_device = self._get_best_output_device()
        self.sample_rate = self._get_compatible_sample_rate(self.input_device)
        print(f"🎯 音频设备预选择完成:")
        print(f"   输入设备: {self.input_device}")
        print(f"   输出设备: {self.output_device}")
        print(f"   采样率: {self.sample_rate}Hz")
    
    def _get_compatible_sample_rate(self, input_device_index):
        """
        检测并选择兼容的采样率
        优先选择WebRTC VAD支持的采样率：8000, 16000, 32000, 48000 Hz
        """
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            
            try:
                # WebRTC VAD 支持的采样率，按优先级排序
                vad_supported_rates = [16000, 48000, 32000, 8000]
                
                # 使用传入的设备索引，如果为None则使用默认设备
                if input_device_index is None:
                    try:
                        device_index = pa.get_default_input_device_info()['index']
                    except:
                        device_index = 0
                else:
                    device_index = input_device_index
                
                print(f"🔍 检测设备 {device_index} 的采样率支持情况...")
                
                # 测试每个采样率是否被设备支持
                for rate in vad_supported_rates:
                    try:
                        # 使用 PyAudio 的 is_format_supported 检测
                        if pa.is_format_supported(
                            rate=rate,
                            input_device=device_index,
                            input_channels=1,
                            input_format=pyaudio.paInt16
                        ):
                            print(f"✅ 选择采样率: {rate} Hz (WebRTC VAD兼容)")
                            return rate
                    except ValueError:
                        # 不支持该采样率
                        continue
                    except Exception as e:
                        # 其他错误，继续尝试下一个
                        print(f"⚠️ 测试采样率 {rate} Hz 时出错: {e}")
                        continue
                
                # 如果所有WebRTC VAD采样率都不支持，尝试获取设备默认采样率
                try:
                    device_info = pa.get_device_info_by_index(device_index)
                    device_rate = int(device_info['defaultSampleRate'])
                    print(f"⚠️ WebRTC VAD采样率均不支持，设备默认采样率: {device_rate} Hz")
                    
                    # 选择最接近的WebRTC VAD支持的采样率
                    closest_rate = min(vad_supported_rates, key=lambda x: abs(x - device_rate))
                    print(f"📍 选择最接近的VAD支持采样率: {closest_rate} Hz")
                    return closest_rate
                    
                except Exception as e:
                    print(f"⚠️ 获取设备信息失败: {e}")
                
            finally:
                pa.terminate()
            
            # 最后的备选方案
            print("⚠️ 采样率检测失败，使用默认值: 16000 Hz")
            return 16000
            
        except ImportError:
            print("⚠️ PyAudio 未安装，使用默认采样率: 16000 Hz")
            return 16000
        except Exception as e:
            print(f"⚠️ 采样率检测异常: {e}，使用默认值: 16000 Hz")
            return 16000

        
    def record_audio(self, duration=None, use_vad=True):
        """录制音频，返回numpy数组"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # 使用预选择的输入设备
            input_device = self.input_device
            
            if duration:
                # 固定时长录音
                print(f"🎤 开始录音 {duration} 秒...")
                recording = sd.rec(
                    int(duration * self.sample_rate), 
                    samplerate=self.sample_rate, 
                    channels=self.channels,
                    dtype=self.dtype,
                    device=input_device
                )
                sd.wait()
                return recording
            else:
                # 手动停止录音
                print("🎤 开始录音... 按回车键停止")
                frames = []
                
                def callback(indata, frames_count, time, status):
                    if status:
                        print(f"⚠️ 录音状态: {status}")
                    frames.append(indata.copy())
                
                try:
                    with sd.InputStream(callback=callback, 
                                      samplerate=self.sample_rate, 
                                      channels=self.channels,
                                      dtype=self.dtype,
                                      device=input_device):
                        try:
                            input()  # 等待用户按回车
                        except KeyboardInterrupt:
                            print("\n🛑 用户中断录音")
                            raise
                    
                    if frames:
                        return np.concatenate(frames, axis=0)
                    return None
                except Exception as stream_error:
                    print(f"录音流错误: {stream_error}")
                    # 尝试回退方案
                    return self._fallback_record()
                
        except Exception as e:
            print(f"sounddevice录音失败: {e}")
            # 尝试回退到pyaudio
            return self._fallback_record()
    
    def _get_best_input_device(self):
        """获取最佳输入设备"""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device))
            
            if not input_devices:
                return None  # 使用默认设备
            
            # 优先选择 DJI 麦克风
            for idx, d in input_devices:
                name = d['name'].lower()
                print(f"🎤 检测设备: {d['name']}")
                if 'dji' in name:
                    print(f"🎤 优先选择 DJI 麦克风: {d['name']} (设备 {idx})")
                    return idx

            # 优先选择USB设备
            for device_idx, device in input_devices:
                device_name = device['name'].lower()
                if any(usb_indicator in device_name for usb_indicator in 
                      ['Wireless','MC87', 'usb', 'headset', 'microphone', 'webcam']):
                    print(f"🎯 选择USB设备: {device['name']}")
                    return device_idx
            
            # 选择第一个可用设备
            device_idx, device = input_devices[0]
            print(f"🎤 选择设备: {device['name']}")
            return device_idx
            
        except Exception as e:
            print(f"设备选择失败: {e}")
            return None
    
    def _fallback_record(self):
        """回退到pyaudio录音"""
        try:
            print("🔄 尝试pyaudio回退录音...")
            import pyaudio
            import numpy as np
            
            p = pyaudio.PyAudio()
            
            # 使用16000Hz作为回退采样率
            fallback_rate = 16000
            chunk = 1024
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=fallback_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            print("🎤 PyAudio录音... 按回车键停止")
            frames = []
            
            def record_thread():
                while True:
                    try:
                        data = stream.read(chunk, exception_on_overflow=False)
                        frames.append(data)
                    except:
                        break
            
            import threading
            thread = threading.Thread(target=record_thread)
            thread.daemon = True
            thread.start()
            
            input()  # 等待用户按回车
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if frames:
                # 转换为numpy数组
                audio_data = b''.join(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # 重塑为sounddevice兼容格式
                audio_array = audio_array.reshape(-1, 1).astype(np.int16)
                
                # 更新采样率
                self.sample_rate = fallback_rate
                print(f"✅ PyAudio录音成功，采样率: {fallback_rate}Hz")
                return audio_array
            
            return None
            
        except Exception as e:
            print(f"PyAudio回退录音也失败: {e}")
            return None
    
    def save_audio(self, audio_data, filename):
        """保存音频到文件"""
        try:
            import soundfile as sf
            
            # 确保音频数据格式正确
            if audio_data is None:
                print("⚠️ 音频数据为空")
                return False
            
            # 如果是2D数组，确保是正确的形状
            if len(audio_data.shape) > 1 and audio_data.shape[1] == 1:
                audio_data = audio_data.flatten()
            
            sf.write(filename, audio_data, self.sample_rate)
            print(f"✅ 音频已保存: {filename} ({self.sample_rate}Hz)")
            return True
        except Exception as e:
            print(f"soundfile保存失败: {e}")
            # 回退到wave保存
            return self._fallback_save(audio_data, filename)
    
    def _fallback_save(self, audio_data, filename):
        """回退到wave保存音频"""
        try:
            import wave
            import numpy as np
            
            # 确保数据是int16格式
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16位
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            print(f"✅ 音频已保存(wave): {filename}")
            return True
        except Exception as e:
            print(f"wave保存也失败: {e}")
            return False
    
    def play_audio(self, filename):
        """播放音频文件 - 智能采样率处理"""
        try:
            import sounddevice as sd
            import soundfile as sf
            
            data, file_sample_rate = sf.read(filename)
            print(f"🔊 播放音频: {filename}")
            
            # 使用预选择的输出设备
            output_device = self.output_device
            
            # 直接使用设备采样率播放，避免报错
            target_rate = self.sample_rate  # 使用初始化时检测到的采样率
            
            if file_sample_rate != target_rate:
                print(f"🔄 转换采样率: {file_sample_rate}Hz → {target_rate}Hz")
                # 简单重采样
                import numpy as np
                ratio = target_rate / file_sample_rate
                new_length = int(len(data) * ratio)
                resampled_data = np.interp(
                    np.linspace(0, len(data), new_length),
                    np.arange(len(data)),
                    data
                )
                data = resampled_data
                file_sample_rate = target_rate
            
            # 直接播放
            sd.play(data, file_sample_rate, device=output_device)
            sd.wait()
            print(f"✅ 播放成功: {file_sample_rate}Hz")
            return True
                
        except Exception as e:
            print(f"播放音频失败: {e}")
            # 回退到系统播放
            return self._fallback_play_system(filename)
    
    def _get_best_output_device(self):
        """获取最佳输出设备"""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            output_devices = []
            
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device))
            
            if not output_devices:
                return None  # 使用默认设备
            
            # 优先选择USB设备
            for device_idx, device in output_devices:
                device_name = device['name'].lower()
                print(f"🎤 检测设备: {device_name}")
                if any(usb_indicator in device_name for usb_indicator in 
                      ['5352', '3293','earpod', 'stereo', 'xiaomi', 'headset', 'headphone', 'usb']):
                    print(f"🔊 选择USB输出设备: {device['name']}")
                    return device_idx
            
            # 选择第一个可用设备
            device_idx, device = output_devices[0]
            print(f"🔊 选择输出设备: {device['name']}")
            return device_idx
            
        except Exception as e:
            print(f"输出设备选择失败: {e}")
            return None
    
    def _play_with_resampling(self, data, original_rate, output_device):
        """重采样后播放音频"""
        try:
            import sounddevice as sd
            
            # 测试不同采样率
            test_rates = [48000, 44100, 22050, 16000]
            
            for target_rate in test_rates:
                if target_rate == original_rate:
                    continue
                    
                try:
                    print(f"🔄 尝试重采样到 {target_rate}Hz...")
                    
                    # 简单的重采样（线性插值）
                    import numpy as np
                    ratio = target_rate / original_rate
                    
                    if ratio != 1.0:
                        # 重采样
                        new_length = int(len(data) * ratio)
                        resampled_data = np.interp(
                            np.linspace(0, len(data), new_length),
                            np.arange(len(data)),
                            data
                        )
                    else:
                        resampled_data = data
                    
                    # 尝试播放
                    sd.play(resampled_data, target_rate, device=output_device)
                    sd.wait()
                    print(f"✅ 重采样播放成功: {target_rate}Hz")
                    return True
                    
                except Exception as resample_error:
                    print(f"⚠️ {target_rate}Hz 重采样失败: {resample_error}")
                    continue
            
            # 所有采样率都失败，尝试回退到系统播放
            return self._fallback_play(data, original_rate)
            
        except Exception as e:
            print(f"重采样播放失败: {e}")
            return False
    
    def _fallback_play(self, data, sample_rate):
        """回退播放方案"""
        try:
            print("🔄 尝试系统播放命令...")
            import tempfile
            import subprocess
            import soundfile as sf
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_filename = tmp_file.name
            
            sf.write(temp_filename, data, sample_rate)
            
            # 尝试系统播放命令
            play_commands = [
                ['aplay', temp_filename],
                ['paplay', temp_filename],
                ['play', temp_filename]
            ]
            
            for cmd in play_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"✅ 系统播放成功: {' '.join(cmd)}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("❌ 所有播放方案都失败")
            return False
            
        except Exception as e:
            print(f"回退播放失败: {e}")
            return False
    
    def _fallback_play_system(self, filename):
        """系统播放回退方案"""
        try:
            import subprocess
            
            # 尝试系统播放命令
            play_commands = [
                ['aplay', filename],
                ['paplay', filename],
                ['play', filename]
            ]
            
            for cmd in play_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        print(f"✅ 系统播放成功: {' '.join(cmd)}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("❌ 系统播放也失败")
            return False
            
        except Exception as e:
            print(f"系统播放失败: {e}")
            return False
    
    def convert_audio_format(self, input_file, output_file, target_sample_rate=None, target_channels=None):
        """转换音频格式"""
        try:
            import soundfile as sf
            import numpy as np
            
            data, sample_rate = sf.read(input_file)
            
            # # 转换采样率
            # if target_sample_rate and target_sample_rate != sample_rate:
            #     try:
            #         import librosa
            #         data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)
            #         sample_rate = target_sample_rate
            #     except ImportError:
            #         print("⚠️ librosa未安装，跳过采样率转换")
            
            # 转换声道数
            if target_channels:
                if target_channels == 1 and len(data.shape) > 1:
                    # 转为单声道
                    data = data.mean(axis=1)
                elif target_channels == 2 and len(data.shape) == 1:
                    # 转为立体声
                    data = np.stack([data, data], axis=1)
            
            sf.write(output_file, data, sample_rate)
            return True
        except Exception as e:
            print(f"音频格式转换失败: {e}")
            return False
    
    def _force_reinitialize_audio_system(self):
        """强力重新初始化音频系统 - 解决长时间运行后的设备状态问题"""
        try:
            import sounddevice as sd
            import gc
            
            print("🔄 强力重新初始化音频系统...")
            
            # 1. 清除sounddevice的内部缓存
            try:
                # 强制查询设备以刷新内部状态  
                _ = sd.query_devices()
                print("   ✅ sounddevice设备缓存已刷新")
            except Exception as e:
                print(f"   ⚠️ 设备缓存刷新失败: {e}")
            
            # 2. 重新检测设备
            old_input = self.input_device
            old_output = self.output_device
            
            self.input_device = self._get_best_input_device()
            self.output_device = self._get_best_output_device()
            
            # 3. 验证设备可用性
            if self.input_device != old_input:
                print(f"   🔄 输入设备已更新: {old_input} → {self.input_device}")
            
            if self.output_device != old_output:
                print(f"   🔄 输出设备已更新: {old_output} → {self.output_device}")
            
            # 4. 测试新设备
            success = self._test_audio_device_health()
            
            if success:
                print("   ✅ 音频系统重新初始化成功")
                return True
            else:
                print("   ⚠️ 音频系统重新初始化后仍有问题")
                return False
                
        except Exception as e:
            print(f"   ❌ 音频系统重新初始化失败: {e}")
            return False
    
    def _test_audio_device_health(self):
        """测试音频设备健康状态"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # 短暂测试录音功能
            test_duration = 0.1  # 100ms
            try:
                recording = sd.rec(
                    int(test_duration * self.sample_rate), 
                    samplerate=self.sample_rate, 
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.input_device
                )
                sd.wait()
                
                # 检查录音数据是否有效
                if recording is not None and len(recording) > 0:
                    print(f"   ✅ 设备健康检查通过: {len(recording)} 样本")
                    return True
                else:
                    print("   ❌ 设备健康检查失败: 无录音数据")
                    return False
                    
            except Exception as test_error:
                print(f"   ❌ 设备健康检查失败: {test_error}")
                return False
                
        except Exception as e:
            print(f"   ❌ 设备健康检查异常: {e}")
            return False


class SimplifiedVoiceRecorder:
    """简化的语音录制器 - 跨平台版本，支持高级VAD"""
    
    def __init__(self):
        self.audio_manager = CrossPlatformAudioManager()
        self.is_recording = False
        self.is_listening = False
        self.audio_data = None
        self.last_activity_time = time.time()
        self.activity_timeout = 15.0
        self._vad_prewarmed = False
        # 键盘监听相关
        self._space_pressed = False
        self._keyboard_listener_thread = None
        self._stop_keyboard_listener = threading.Event()
        try:
            self.advanced_vad = AdvancedVoiceActivityDetector(
                sample_rate=self.audio_manager.sample_rate,
                buffer_duration=10.0,      # 4秒环形缓冲区（增加容量）
                lookback_duration=1.5,    # 1000ms回溯（增加回溯时间）
                silence_timeout=1.0,      # 2.0秒静音超时（更快结束）
                min_speech_duration=0.4,  # 最小语音400ms（更短语音）
                speech_threshold=0.3,     # VAD阈值（更敏感）
                overlap_threshold=0.3      # 重合率阈值（更快响应）
            )
            self.vad_enabled = True
            if self.advanced_vad.vad_available:
                print("✅ 高级VAD语音活动检测已启用（Silero VAD优化版）")
                print("💡 音频累积+重采样+环形缓冲区：完美解决背景噪声问题")
            else:
                print("✅ 高级VAD语音活动检测已启用（环形缓冲区+基础VAD模式）")
                print("💡 核心功能：持续监听+按需录音+智能回溯")
        except Exception as e:
            print(f"⚠️ 高级VAD初始化失败: {e}")
            self.vad = VoiceActivityDetector()
            self.advanced_vad = None
            self.vad_enabled = True
            print("⚠️ 使用传统基础VAD")
        # 初始化键盘监听
        self._init_keyboard_listener()

    def _init_keyboard_listener(self):
        def listen():
            try:
                from pynput import keyboard
            except ImportError as e:
                if "X connection" in str(e) or "DISPLAY" in str(e):
                    print(f"⚠️ 当前环境不支持 pynput (无 X11/DISPLAY)，跳过空格键录音监听: {e}")
                    return
                print("⚠️ 未检测到 pynput 库或导入失败，自动安装...")
                os.system('pip install pynput')
                try:
                    from pynput import keyboard
                except Exception as e:
                    print(f"⚠️ 安装后仍然无法导入 pynput，跳过空格键监听: {e}")
                    return
            except Exception as e:
                print(f"⚠️ pynput 导入异常，跳过空格键录音监听: {e}")
                return
            def on_press(key):
                if key == keyboard.Key.space:
                    self._space_pressed = True
            def on_release(key):
                if key == keyboard.Key.space:
                    self._space_pressed = False
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            while not self._stop_keyboard_listener.is_set():
                time.sleep(0.05)
            listener.stop()
        self._keyboard_listener_thread = threading.Thread(target=listen, daemon=True)
        self._keyboard_listener_thread.start()
        
    def _check_and_reset_if_inactive(self):
        """检查是否长时间不活跃，如果是则重置VAD状态"""
        current_time = time.time()
        if current_time - self.last_activity_time > self.activity_timeout:
            print(f"⏰ 检测到{self.activity_timeout}秒无活动，自动重置VAD状态...")
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.cleanup()
            # self.last_activity_time = current_time
            # print("🔄 VAD状态重置完成，准备接收新的语音输入")
            print("🔄 VAD重置功能已暂时禁用，仅更新活动时间")
            self.last_activity_time = current_time
    
    def _check_audio_device_health(self):
        """检查音频设备健康状态，如果异常则重新初始化"""
        try:
            # 检查音频设备是否可用
            if not hasattr(self.audio_manager, '_test_audio_device_health'):
                return True  # 如果没有健康检查方法，假设设备正常
            
            device_healthy = self.audio_manager._test_audio_device_health()
            
            if not device_healthy:
                print("⚠️ 音频设备健康检查失败，尝试重新初始化...")
                
                # 尝试重新初始化音频管理器
                try:
                    # 重新获取最佳输入设备
                    new_input_device = self.audio_manager._get_best_input_device()
                    if new_input_device is not None:
                        self.audio_manager.input_device = new_input_device
                        print(f"🔄 音频设备重新初始化成功: {new_input_device}")
                        return True
                    else:
                        print("❌ 无法重新获取音频设备")
                        return False
                except Exception as e:
                    print(f"❌ 音频设备重新初始化失败: {e}")
                    return False
            else:
                return True
                
        except Exception as e:
            print(f"⚠️ 音频设备健康检查异常: {e}")
            return False
        
    def start_recording(self, use_vad=True):
        """开始录音 - 支持高级VAD自动停止（优化版，支持预热后快速启动）"""
        # 检查并处理长时间不活跃的情况
        self._check_and_reset_if_inactive()
        
        # 快速音频设备健康检查（如果已经预热过，跳过详细检查）
        if hasattr(self, '_vad_prewarmed') and self._vad_prewarmed:
            print("⚡ VAD已预热，快速启动录音...")
            self._vad_prewarmed = False  # 重置预热标志
            self.last_activity_time = time.time()
            return self._record_with_advanced_vad() if use_vad else self._record_with_basic_vad()
        
        # 音频设备健康检查
        if not self._check_audio_device_health():
            print("⚠️ 音频设备健康检查失败，尝试使用基础录音模式...")
            # 如果健康检查失败，尝试基础录音
            try:
                self.is_recording = True
                self.audio_data = self.audio_manager.record_audio(duration=5.0)  # 5秒测试录音
                self.is_recording = False
                if self.audio_data is not None:
                    print("✅ 基础录音测试成功，继续正常流程")
                    self.last_activity_time = time.time()
                    return True
                else:
                    print("❌ 基础录音测试失败，无法继续")
                    return False
            except Exception as e:
                print(f"❌ 基础录音测试异常: {e}")
                return False
        
        # 更新活动时间
        self.last_activity_time = time.time()
        
        # 系统准备就绪，结束计时（如果不是预热模式）
        if not (hasattr(self, '_vad_prewarmed') and self._vad_prewarmed):
            # 通知主控制器系统准备就绪
            if hasattr(self, 'controller') and self.controller:
                self.controller._end_response_cycle_timing()
        
        if not use_vad or not self.vad_enabled:
            # 手动模式
            self.is_recording = True
            self.audio_data = self.audio_manager.record_audio()
            self.is_recording = False
            return self.audio_data is not None
        else:
            # 高级VAD自动模式
            if self.advanced_vad:
                result = self._record_with_advanced_vad()
            else:
                result = self._record_with_basic_vad()
            
            # 录音完成后更新活动时间
            if result:
                self.last_activity_time = time.time()
                
            return result
    
    def _record_with_advanced_vad(self):
        """使用高级VAD进行智能录音 - 仅在空格键按下时录音，松开时停止"""
        try:
            import sounddevice as sd
            import numpy as np
            import threading
            # 健康检查
            if not self._check_audio_device_health():
                print("⚠️ 高级VAD录音前音频设备健康检查失败，尝试重试...")
                time.sleep(1.0)
                if not self._check_audio_device_health():
                    print("❌ 音频设备健康检查持续失败，回退到基础录音")
                    return self._record_with_basic_vad()
            print("🎤 按住空格键开始录音，松开空格键停止录音...")
            input_device = self.audio_manager._get_best_input_device()
            self.advanced_vad.reset()
            self.is_listening = True
            self.is_recording = False
            self.audio_data = None
            recording_complete = threading.Event()
            def audio_callback(indata, frames_count, time_, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                # 只有空格键按下时才录音
                if self._space_pressed:
                    try:
                        if indata.dtype != np.int16:
                            audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                        else:
                            audio_int16 = indata.flatten().astype(np.int16)
                        vad_result = self.advanced_vad.feed_audio(audio_int16)
                        if not self.is_recording:
                            self.is_recording = True
                            print("🔴 开始录音...")
                    except Exception as vad_error:
                        print(f"⚠️ VAD检测失败: {vad_error}")
                else:
                    # 松开空格键，强制结束录音
                    if self.is_recording:
                        try:
                            speech_audio = self.advanced_vad.get_speech_audio()
                            if speech_audio is not None and len(speech_audio) > 0:
                                self.audio_data = speech_audio
                                print(f"✅ 录音完成: {len(self.audio_data)}样本")
                            else:
                                print("⚠️ 未能获取语音数据")
                                self.audio_data = None
                        except Exception as audio_error:
                            print(f"⚠️ 获取语音数据失败: {audio_error}")
                            self.audio_data = None
                        self.is_recording = False
                        self.is_listening = False
                        recording_complete.set()
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.audio_manager.sample_rate,
                    channels=self.audio_manager.channels,
                    dtype=self.audio_manager.dtype,
                    device=input_device,
                    blocksize=512
                ):
                    print("👂 正在监听空格键... (按空格录音，松开停止)")
                    idle_start_time = time.time()
                    while not recording_complete.wait(timeout=0.1):
                        current_time = time.time()
                        self._check_silero_vad_health_periodically()
                        # 超时自动退出
                        if current_time - idle_start_time > 60.0:
                            print("⏰ 长时间无操作，自动结束监听...")
                            self.is_listening = False
                            self.is_recording = False
                            recording_complete.set()
                            break
                        if self.is_recording:
                            idle_start_time = current_time
            except KeyboardInterrupt:
                print("\n🛑 用户中断录音")
                self.is_listening = False
                self.is_recording = False
                recording_complete.set()
                raise
            except Exception as stream_error:
                print(f"❌ 音频流错误: {stream_error}")
                return False
            print("🏁 智能录音完成")
            return self.audio_data is not None
        except Exception as e:
            print(f"❌ 高级VAD录音失败: {e}")
            print("🔧 尝试重新初始化音频系统来解决问题...")
            if hasattr(self.audio_manager, '_force_reinitialize_audio_system'):
                try:
                    reinit_success = self.audio_manager._force_reinitialize_audio_system()
                    if reinit_success:
                        print("✅ 音频系统重新初始化成功，重试高级VAD录音...")
                        try:
                            return self._record_with_advanced_vad_retry()
                        except Exception as retry_error:
                            print(f"⚠️ 重试高级VAD录音仍失败: {retry_error}")
                    else:
                        print("⚠️ 音频系统重新初始化失败")
                except Exception as reinit_error:
                    print(f"⚠️ 音频系统重新初始化异常: {reinit_error}")
            print("🔄 回退到基础录音模式...")
            return self._record_with_basic_vad()
    
    def _record_with_basic_vad(self):
        """使用基础VAD进行录音（仅空格键按下时录音，松开时停止）"""
        try:
            import sounddevice as sd
            import numpy as np
            if not self._check_audio_device_health():
                print("⚠️ 基础VAD录音前音频设备健康检查失败，尝试重试...")
                time.sleep(1.0)
                if not self._check_audio_device_health():
                    print("❌ 音频设备健康检查持续失败，回退到手动录音")
                    self.is_recording = True
                    self.audio_data = self.audio_manager.record_audio()
                    self.is_recording = False
                    return self.audio_data is not None
            print("🎤 按住空格键开始录音，松开空格键停止录音...")
            input_device = self.audio_manager._get_best_input_device()
            if self.vad:
                self.vad.reset()
            frames = []
            self.is_recording = False
            recording_complete = threading.Event()
            def audio_callback(indata, frames_count, time_, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                if self._space_pressed:
                    frames.append(indata.copy())
                    if not self.is_recording:
                        self.is_recording = True
                        print("🔴 开始录音...")
                else:
                    if self.is_recording:
                        self.is_recording = False
                        if frames:
                            self.audio_data = np.concatenate(frames, axis=0)
                            print(f"✅ 基础VAD录音完成: {self.audio_data.shape}")
                        else:
                            print("❌ 没有录制到音频数据")
                            self.audio_data = None
                        recording_complete.set()
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.audio_manager.sample_rate,
                    channels=self.audio_manager.channels,
                    dtype=self.audio_manager.dtype,
                    device=input_device
                ):
                    print("👂 正在监听空格键... (按空格录音，松开停止)")
                    idle_start_time = time.time()
                    while not recording_complete.wait(timeout=0.1):
                        current_time = time.time()
                        if current_time - idle_start_time > 60.0:
                            print("⏰ 长时间无操作，自动结束监听...")
                            recording_complete.set()
                            break
                        if self.is_recording:
                            idle_start_time = current_time
            except KeyboardInterrupt:
                print("\n🛑 用户中断，停止基础VAD录音")
                self.is_recording = False
                recording_complete.set()
                raise
            if self.audio_data is not None:
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ 基础VAD录音失败: {e}")
            print("🔄 回退到手动录音模式...")
            self.is_recording = True
            self.audio_data = self.audio_manager.record_audio()
            self.is_recording = False
            return self.audio_data is not None
    
    def save_audio(self, filename):
        """保存录制的音频"""
        if self.audio_data is not None:
            return self.audio_manager.save_audio(self.audio_data, filename)
        return False
    
    def cleanup(self):
        """清理资源 - 增强版，包含音频系统重新初始化和Silero VAD健康检测"""
        self.audio_data = None
        self.is_recording = False
        self.is_listening = False
        
        # 重置活动时间
        self.last_activity_time = time.time()
        
        # 彻底重置VAD状态
        if self.advanced_vad:
            # 先进行Silero VAD健康检测
            if hasattr(self.advanced_vad, 'silero_vad') and self.advanced_vad.silero_vad:
                try:
                    health_ok = self.advanced_vad.silero_vad.health_check()
                    if not health_ok:
                        print("🔧 检测到Silero VAD状态异常，执行深度重置...")
                        # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                        # self.advanced_vad.silero_vad.reset_buffer()
                        # print("✅ 已执行轻量级重置（避免影响音频播放）")
                        print("✅ VAD重置功能已暂时禁用，仅记录状态异常")
                    else:
                        print("✅ Silero VAD状态正常")
                except Exception as e:
                    print(f"⚠️ Silero VAD健康检测异常: {e}")
            
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.advanced_vad.reset()
            print("🔄 VAD重置功能已暂时禁用")
        elif self.vad:
            # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
            # self.vad.reset()
            print("🔄 基础VAD重置功能已暂时禁用")
        
        # 强力重新初始化音频系统（解决长时间运行后的设备问题）
        if hasattr(self.audio_manager, '_force_reinitialize_audio_system'):
            try:
                reinit_success = self.audio_manager._force_reinitialize_audio_system()
                if reinit_success:
                    print("🔄 音频系统强力重新初始化成功")
                else:
                    print("⚠️ 音频系统重新初始化有问题，但会继续尝试")
            except Exception as e:
                print(f"⚠️ 音频系统重新初始化异常: {e}")
                
        print("✅ SimplifiedVoiceRecorder资源清理完成（增强版）")

    def _check_silero_vad_health_periodically(self):
        """定期检查Silero VAD健康状态"""
        if not self.advanced_vad or not hasattr(self.advanced_vad, 'silero_vad'):
            return
        
        try:
            current_time = time.time()
            # 每30秒检查一次Silero VAD健康状态
            if not hasattr(self, '_last_silero_health_check'):
                self._last_silero_health_check = 0
            
            if current_time - self._last_silero_health_check > 30.0:
                self._last_silero_health_check = current_time
                
                # 检查是否正在播放音频，如果是则跳过健康检查
                if hasattr(self, 'audio_manager') and hasattr(self.audio_manager, '_play_lock'):
                    if self.audio_manager._play_lock.locked():
                        print("🎵 检测到音频正在播放，跳过Silero VAD健康检查")
                        return
                
                if self.advanced_vad.silero_vad:
                    health_ok = self.advanced_vad.silero_vad.health_check()
                    if not health_ok:
                        print("🔧 定期检测到Silero VAD状态异常，执行自动修复...")
                        # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                        # self.advanced_vad.silero_vad.reset_buffer()
                        # print("✅ 已执行轻量级重置（避免影响音频播放）")
                        print("✅ VAD重置功能已暂时禁用，仅记录状态异常")
                        
        except Exception as e:
            print(f"⚠️ 定期Silero VAD健康检测异常: {e}")

    def _record_with_advanced_vad_retry(self):
        """高级VAD录音重试方法 - 简化版，避免无限递归"""
        try:
            import sounddevice as sd
            import numpy as np
            import threading
            
            print("🔄 重试高级VAD录音（简化版）...")
            
            # 获取输入设备（重新获取）
            input_device = self.audio_manager._get_best_input_device()
            if input_device is None:
                print("❌ 无法获取有效输入设备")
                return False
            
            # 重置VAD状态
            if self.advanced_vad:
                # 暂时禁用VAD重置功能，测试是否与长时间不活跃问题相关
                # self.advanced_vad.reset()
                print("🔄 VAD重置功能已暂时禁用（重试模式）")
            
            # 状态管理
            self.is_listening = True
            self.is_recording = False
            self.audio_data = None
            recording_complete = threading.Event()
            
            def audio_callback(indata, frames_count, time, status):
                if status:
                    print(f"⚠️ 录音状态: {status}")
                
                if not self.is_listening:
                    return
                
                # 将音频数据输入到高级VAD
                try:
                    if indata.dtype != np.int16:
                        audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                    else:
                        audio_int16 = indata.flatten().astype(np.int16)
                    
                    vad_result = self.advanced_vad.feed_audio(audio_int16)
                    
                    if vad_result == "speech_start":
                        if not self.is_recording:
                            self.is_recording = True
                            print("🔴 重试录音开始...")
                    
                    elif vad_result == "speech_end":
                        if self.is_recording:
                            try:
                                speech_audio = self.advanced_vad.get_speech_audio()
                                if speech_audio is not None and len(speech_audio) > 0:
                                    self.audio_data = speech_audio
                                    print(f"✅ 重试录音完成: {len(speech_audio)}样本")
                                else:
                                    print("⚠️ 重试录音未能获取语音数据")
                                    self.audio_data = None
                            except Exception as audio_error:
                                print(f"⚠️ 重试录音数据获取失败: {audio_error}")
                                self.audio_data = None
                            
                            self.is_recording = False
                            self.is_listening = False
                            recording_complete.set()
                    
                except Exception as e:
                    print(f"⚠️ 重试VAD处理音频时出错: {e}")
                    self.is_listening = False
                    self.is_recording = False
                    recording_complete.set()
            
            # 开始音频流
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.audio_manager.sample_rate,
                    channels=self.audio_manager.channels,
                    dtype=self.audio_manager.dtype,
                    device=input_device,
                    blocksize=512
                ):
                    print("👂 重试监听中... (最多15秒)")
                    
                    # 等待录音完成，但限制最大等待时间
                    if recording_complete.wait(timeout=15.0):
                        print("🏁 重试录音完成")
                    else:
                        print("⏰ 重试录音超时")
                        self.is_listening = False
                        self.is_recording = False
                    
            except Exception as stream_error:
                print(f"❌ 重试音频流错误: {stream_error}")
                return False
            
            return self.audio_data is not None
                
        except Exception as e:
            print(f"❌ 重试高级VAD录音失败: {e}")
            return False



class SimplifiedAudioPlayer:
    """简化的音频播放器 - 跨平台版本，使用专用进程维护音频播放"""
    
    def __init__(self):
        self.audio_manager = CrossPlatformAudioManager()
        self._play_queue = None  # 播放队列
        self._control_queue = None  # 控制队列
        self._audio_process = None  # 音频播放进程
        self._start_audio_process()  # 启动音频播放进程


    def get_audio_duration(self, audio_file):
        """获取音频文件时长（秒）"""
        if not audio_file or not os.path.exists(audio_file):
            return 0.0
        
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_file)
            duration = len(data) / samplerate
            return duration
        except Exception as e:
            print(f"⚠️ 无法获取音频时长: {e}")
            return 0.0

    def _start_audio_process(self):
        """启动专用的音频播放进程"""
        try:
            import multiprocessing as mp
            
            # 创建进程间通信队列
            self._play_queue = mp.Queue()  # 播放请求队列
            self._control_queue = mp.Queue()  # 控制命令队列
            
            # 启动音频播放进程
            self._audio_process = mp.Process(
                target=self._audio_worker_process,
                args=(self._play_queue, self._control_queue),
                daemon=True
            )
            self._audio_process.start()
            
            print(f"🎵 音频播放进程已启动 (PID: {self._audio_process.pid})")
            
        except Exception as e:
            print(f"❌ 启动音频播放进程失败: {e}")
            self._audio_process = None
    
    def _audio_worker_process(self, play_queue, control_queue):
        """音频播放工作进程 - 在独立进程中运行"""
        try:
            # 在进程中创建音频管理器
            process_audio_manager = CrossPlatformAudioManager()
            
            print(f"🎵 音频播放进程 {os.getpid()} 已就绪，等待播放请求...")
            
            while True:
                try:
                    # 检查控制命令
                    try:
                        while not control_queue.empty():
                            command = control_queue.get_nowait()
                            if command == "STOP":
                                print("⏹️ 收到停止命令，音频播放进程退出")
                                return
                            elif command == "PAUSE":
                                print("⏸️ 收到暂停命令")
                                # 这里可以添加暂停逻辑
                    except:
                        pass
                    
                    # 检查播放请求
                    try:
                        if not play_queue.empty():
                            audio_file = play_queue.get_nowait()
                            print(f"🎵 进程 {os.getpid()} 开始播放: {audio_file}")
                            
                            # 播放音频
                            success = process_audio_manager.play_audio(audio_file)
                            
                            if success:
                                print(f"✅ 音频播放完成: {audio_file}")
                            else:
                                print(f"❌ 音频播放失败: {audio_file}")
                        else:
                            # 没有播放请求时短暂休眠
                            import time
                            time.sleep(0.1)
                    except:
                        pass
                        
                except KeyboardInterrupt:
                    print("🛑 音频播放进程收到中断信号")
                    break
                except Exception as e:
                    print(f"⚠️ 音频播放进程异常: {e}")
                    import time
                    time.sleep(1.0)  # 异常后等待1秒再继续
            
        except Exception as e:
            print(f"❌ 音频播放进程严重错误: {e}")
        finally:
            print(f"🏁 音频播放进程 {os.getpid()} 退出")
    
    def play_audio_file(self, audio_file):
        """播放音频文件 - 向专用进程发送播放请求"""
        if not audio_file or not os.path.exists(audio_file):
            print("⚠️ 音频文件不存在，跳过播放")
            return False
        
        if not self._audio_process or not self._audio_process.is_alive():
            print("⚠️ 音频播放进程未运行，尝试重新启动...")
            self._start_audio_process()
            if not self._audio_process or not self._audio_process.is_alive():
                print("❌ 无法启动音频播放进程")
                return False
        
        try:
            # 向播放队列发送音频文件路径
            self._play_queue.put(audio_file)
            print(f"📤 已发送播放请求: {audio_file}")
            return True
            
        except Exception as e:
            print(f"❌ 发送播放请求失败: {e}")
            return False
    
    def stop_current_playback(self):
        """停止当前播放"""
        if self._control_queue:
            try:
                self._control_queue.put("STOP")
                print("⏹️ 已发送停止命令")
            except Exception as e:
                print(f"❌ 发送停止命令失败: {e}")
    
    def pause_playback(self):
        """暂停播放"""
        if self._control_queue:
            try:
                self._control_queue.put("PAUSE")
                print("⏸️ 已发送暂停命令")
            except Exception as e:
                print(f"❌ 发送暂停命令失败: {e}")
    
    def is_playing(self):
        """检查播放进程是否运行"""
        return self._audio_process is not None and self._audio_process.is_alive()
    
    def cleanup(self):
        """清理资源"""
        try:
            if self._control_queue:
                self._control_queue.put("STOP")
            
            if self._audio_process and self._audio_process.is_alive():
                self._audio_process.join(timeout=2.0)
                if self._audio_process.is_alive():
                    self._audio_process.terminate()
                    self._audio_process.join(timeout=1.0)
                    if self._audio_process.is_alive():
                        self._audio_process.kill()
            
            print("✅ 音频播放器资源清理完成")
            
        except Exception as e:
            print(f"⚠️ 清理音频播放器资源时出错: {e}")
    
    def __del__(self):
        """析构函数，确保进程被清理"""
        self.cleanup()




class SimplifiedAudioPlayer_Blocking:
    """简化的音频播放器 - 跨平台版本，线程安全"""
    
    def __init__(self):
        self.audio_manager = CrossPlatformAudioManager()
        self._play_lock = threading.Lock()  # 添加播放锁，防止并发播放冲突
    
    def get_audio_duration(self, audio_file):
        """获取音频文件时长（秒）"""
        if not audio_file or not os.path.exists(audio_file):
            return 0.0
        
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_file)
            duration = len(data) / samplerate
            return duration
        except Exception as e:
            print(f"⚠️ 无法获取音频时长: {e}")
            return 0.0
    
    def play_audio_file(self, audio_file):
        """播放音频文件 - 线程安全版本"""
        if not audio_file or not os.path.exists(audio_file):
            print("⚠️ 音频文件不存在，跳过播放")
            return False
        
        # 使用锁确保同时只有一个音频在播放
        with self._play_lock:
            try:
                return self.audio_manager.play_audio(audio_file)
            except Exception as e:
                print(f"⚠️ 音频播放失败: {e}")
                # 尝试清理可能的资源冲突
                import time
                time.sleep(0.1)  # 短暂等待，让资源释放
                return False
