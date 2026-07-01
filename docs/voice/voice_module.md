# 包内语音转文字模块 (Speech To Text Module)

## 1. 模块作用
本项目从完整的机器人语音交互链路中旁路剥离出一条纯净的“仅语音输入转文字”通道。当前代码已迁入 `agenticlab_human.voice` 包，不再掺杂任何机器人动作库、机器人控制器实例、LLM大模型对话与处理以及TTS语音合成播放机制。

该模块将负责以下完整流水线：
- **麦克风启动与采集**
- **VAD静音/语音活动自动检测（结合缓存回溯）**
- **音频WAV文件处理落地**
- **ASR 调用与文字剥离**
- **临时文件自动清理机制**

对外仅暴露一个简单、稳定、阻塞式的接口 `SpeechInputService.listen() -> str`。只需一次调用便返回识别到的自然语言字符串文本（`user_text`）。它极大地简化了大模型和外部控制器模块的使用门槛。

## 2. 如何运行 Demo
建议先按 [ENVIRONMENT.md](ENVIRONMENT.md) 创建 Python 3.10 环境、安装依赖并完成麦克风测试。在仓库根目录运行：
```bash
python -m agenticlab_human.voice.demo_speech_to_text
```
运行后，程序会提示 `正在等待语音输入 (监听中)...`。当前录音触发方式为**按住空格键开始录音、松开空格键停止并识别**。识别完成后终端会通过形如 `🎯 最终识别结果:【 我要一瓶可乐 】` 输出结果并进入下一轮监听。按 `Ctrl+C` 可安全退出并自动清理产生的录音碎文件。

## 3. 在项目中的复用方式
该模块对外暴露包内服务入口，调用方不需要手动修改 `sys.path`：
```python
from agenticlab_human.voice import SpeechInputService

stt = SpeechInputService(device="cpu")  # 或者 "cuda"
text = stt.listen()
print(f"听到了: {text}")
```

## 4. 依赖包说明 (Python Packages)
本独立模块需确保宿主环境支持下列库：
- `sounddevice`, `soundfile` (录音及音频流支撑)
- `PyAudio` (采样率探测及备用录音方案)
- `numpy` (音频数据计算辅助)
- `pynput` (当前空格键触发录音)
- `torch`, `torchaudio` (Silero VAD 与 ASR 推理依赖)
- `funasr`, `modelscope` (ASR 及官方库相关)

x86 Linux 上建议先安装官方 PyTorch wheel，再安装项目的 `voice` extra：
```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[voice]"
```

## 5. 模型及配置项情况
- **ASR 模型依赖**： `SenseVoiceSmall`，ASR 代码内部会通过 `modelscope` 自动去其默认缓存路径检索加载。如果需要迁移到离线环境，请留意打包 `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall/` 目录。
- **VAD 模型依赖**：当前高级 VAD 会尝试通过 `torch.hub` 加载 Silero VAD；如果失败会回退到基础能量 VAD。离线环境请同步 `~/.cache/torch/hub/snakers4_silero-vad_master/`。
- **自定义配置修改**： 见 `config/speech_config.py`，在这里你能够自定义修改使用的 `DEVICE` 设备名称、临时wav生成的工作目录 `TEMP_AUDIO_DIR` 以及是否启用 `USE_VAD`。更底层如果想改如录音采样率、VAD阈值等，则保留在 `backend/VAD.py` 或 `backend/utils.py` 之内操作。

## 6. 与原有大模型/TTS 的解绑确信
- **不包含大模型**：`listen()` 操作不需载入 `zhipuai`，也完全断绝了向大模型发起请求的功能。
- **不包含 TTS**：移除了需要发送并接收TTS音频的工作步骤。
- **不包含播放器**：无论是原本阻塞的音频播报功能还是作为异步流音频拉取的代码已不在此处干预系统环境。若你看到 `voice/utils.py` 里遗留了 `SimplifiedAudioPlayer` 等类，这只作为物理工具类的连带复制被保存，而我们的 `SpeechInputService` 入口中绝无任何调用的指纹。

---
最后重申：**当前包内模块不对旧项目任何外部环境做出修改破坏**，它仅仅是通过旁路重构，把所需的原材料归档成了高度可复用的输入功能积木。
