"""
本地语音转文字测试脚本
测试音频采集 -> VAD -> ASR 流程，无需云端服务
"""

import numpy as np
import logging
import time
import sys
import os

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_client.audio_buffer import AudioCapture
from local_client.vad_module import VADProcessor
from local_client.asr_engine import SenseVoiceASR, StreamingASR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalASRTest:
    """本地 ASR 测试"""
    
    def __init__(self):
        self.audio_capture = None
        self.vad = None
        self.asr = None
        self.streaming_asr = None
        
        # 语音缓冲
        self.speech_buffer = []
        self.is_speaking = False
        
    def init_modules(self):
        """初始化模块"""
        print("\n" + "="*50)
        print("  初始化语音识别模块...")
        print("="*50 + "\n")
        
        # 1. 初始化音频采集
        print("[1/3] 初始化音频采集...")
        self.audio_capture = AudioCapture()
        print("      音频采集就绪 ✓")
        
        # 2. 初始化 VAD
        print("[2/3] 初始化 VAD (Silero-VAD)...")
        self.vad = VADProcessor(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            use_gpu=True
        )
        print("      VAD 就绪 ✓")
        
        # 3. 初始化 ASR
        print("[3/3] 初始化 ASR (SenseVoiceSmall)...")
        print("      首次运行需要下载模型，请耐心等待...")
        self.asr = SenseVoiceASR(use_gpu=True)
        self.streaming_asr = StreamingASR(
            asr_engine=self.asr,
            on_partial_result=self._on_partial_result,
            on_final_result=self._on_final_result
        )
        print("      ASR 就绪 ✓")
        
        print("\n所有模块初始化完成！\n")
    
    def _on_audio_frame(self, frame: np.ndarray):
        """音频帧回调"""
        # VAD 处理
        result = self.vad.process_frame(frame)
        
        # 如果正在说话，收集音频
        if result.is_speech or self.is_speaking:
            self.speech_buffer.append(frame.copy())
    
    def _on_speech_start(self):
        """语音开始"""
        self.is_speaking = True
        self.speech_buffer = []
        # 获取预缓冲
        pre_buffer = self.audio_capture.get_pre_buffer()
        if len(pre_buffer) > 0:
            self.speech_buffer.append(pre_buffer)
        print("\n🎤 检测到语音，正在录音...", end='', flush=True)
    
    def _on_speech_end(self):
        """语音结束"""
        self.is_speaking = False
        print(" 完成")
        
        if self.speech_buffer:
            # 合并音频
            audio = np.concatenate(self.speech_buffer)
            duration = len(audio) / 16000
            print(f"   录音时长: {duration:.1f}秒")
            
            # 执行识别
            print("   正在识别...", end='', flush=True)
            start_time = time.time()
            result = self.asr.transcribe(audio)
            latency = (time.time() - start_time) * 1000
            
            print(f" 完成 ({latency:.0f}ms)")
            print(f"\n   📝 识别结果: {result.text}\n")
        
        self.speech_buffer = []
    
    def _on_partial_result(self, text: str):
        """部分识别结果"""
        print(f"\r   [部分] {text}", end='', flush=True)
    
    def _on_final_result(self, text: str):
        """最终识别结果"""
        print(f"\n   [最终] {text}")
    
    def run(self):
        """运行测试"""
        try:
            self.init_modules()
            
            print("="*50)
            print("  语音转文字测试")
            print("="*50)
            print("\n说话开始录音，停顿后自动识别")
            print("按 Ctrl+C 退出\n")
            print("-"*50)
            
            # 开始音频采集
            self.audio_capture.start(self._on_audio_frame)
            
            # 主循环
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n正在关闭...")
        finally:
            if self.audio_capture:
                self.audio_capture.stop()
            print("测试结束")


def main():
    test = LocalASRTest()
    test.run()


if __name__ == "__main__":
    main()
