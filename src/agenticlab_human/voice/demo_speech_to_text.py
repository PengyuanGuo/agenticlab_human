import sys
import signal

from agenticlab_human.voice import SpeechInputService

def main():
    service = None

    def _signal_handler(signum, frame):
        print("\n🛑 收到退出信号，正在安全退出...")
        if service:
            service.shutdown()
        sys.exit(0)

    # 注册 Ctrl+C 响应
    signal.signal(signal.SIGINT, _signal_handler)
    
    print("=" * 50)
    print("      🎙️ Speech To Text 独立模块演示       ")
    print("=" * 50)

    try:
        service = SpeechInputService()
        print("✅ 服务启动成功，按 Ctrl+C 退出。\n")
        
        while True:
            # 只专注于提取文字并打印
            text = service.listen()
            if text:
                print(f"🎯 最终识别结果:【 {text} 】\n")
            else:
                print("💤 未获取到有效文本，继续监听...\n")

    except Exception as e:
        print(f"程序启动或者运行严重异常: {e}")
        import traceback
        traceback.print_exc()
        if service:
            service.shutdown()

if __name__ == "__main__":
    main()
