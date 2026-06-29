# 语音转文字包内模块 (STT Module) 环境指南

该模块已精简并旁路解耦，可部署于运行完整版操作系统的上位机开发电脑或基于 Linux/Ubuntu 的边缘盒子 (如 Jetson 工控机) 上。

## 1. 系统要求与工具
- **Python 版本**：推荐 Python 3.10，与仓库 `pyproject.toml` 保持一致
- **底层音频工具**：基于 Linux 环境下，确保已安装 ALSA 驱动体系套件。部分系统可能还需要配置 PulseAudio。
  > 验证麦克风可通过命令排查，如 `arecord -D plughw:CARD=MINI,DEV=0 -f S16_LE -r 16000 -c 1 test.wav`

## 2. 推荐硬件
- **麦克风输入**：默认优先匹配 **DJI Mic Mini** 领夹式麦克风，以保障物理收音和降噪的纯净。若无此设备，代码将按 USB / Default 的顺序自动降级分配默认的板载麦克风。

## 3. 核心模型缓存与离线部署
本模块不直接在代码库中携带 AI 大模型权重参数，但首次执行时将自动于系统缓存中下载。如果在无法联网的 Jetson 物理主机上部署，你需要提前将对应文件夹复制迁移：

- **(1) SenseVoiceSmall 模型 (ASR 使用)**
  - 下载行为：代码借助 `modelscope` 请求云端权重 (`model.pt` 约 800+ MB)。
  - 默认系统缓存路径：`~/.cache/modelscope/hub/models/iic/SenseVoiceSmall/`
- **(2) Silero VAD 模型 (静音检测使用)**
  - 下载行为：通过 `torch.hub.load`。
  - 默认系统缓存路径：`~/.cache/torch/hub/snakers4_silero-vad_master/`

## 4. Jetson 的 Torch 安装警示
代码执行高度依赖 `torch` 做矩阵检测加速。
- **请勿直接 `pip install torch torchaudio`**。
- **正规做法**：在 Nvidia 官方指引页寻找对应当前系统 JetPack 版本 (如 JP 5.1或6.0) 预编译的 `.whl` 文件手动推入。

## 5. Pynput 的 SSH 监听局限
本项目内部仍旧存留小范围的硬断点和 `pynput` 的监听按键介入调试。
- 若通过纯无头 (Headless) SSH 终端连接，没有接入 X11 图形会话 (`DISPLAY`) 时，`pynput` 的监听功能无法拦截到键盘按键事件，可外接显示器或在带 UI 的 Session / Tmux 下运行以获得键盘热键强制中断支持。
