"""
本地客户端主程序
整合音频采集、VAD、ASR、播放和网络通信
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock, Event
from enum import Enum
import logging
import time
import json

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import (
    MessageType,
    AudioChunkHeader,
    AudioConfig,
    TTSConfig,
    NetworkConfig
)
from local_client.audio_buffer import FullDuplexAudio
from local_client.vad_module import VADProcessor, VADResult
from local_client.asr_engine import ASRManager
from local_client.websocket_client import WebSocketClient, AudioReceiver

logger = logging.getLogger(__name__)


class ClientState(Enum):
    """客户端状态"""
    IDLE = "idle"                    # 空闲等待
    LISTENING = "listening"          # 正在监听用户语音
    PROCESSING = "processing"        # 处理中（等待云端响应）
    PLAYING = "playing"              # 播放 AI 响应
    INTERRUPTING = "interrupting"    # 打断处理中


@dataclass
class LatencyMetrics:
    """延迟指标"""
    vad_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    network_rtt_ms: float = 0.0
    total_latency_ms: float = 0.0
    timestamp: float = 0.0


class BargeInController:
    """
    打断控制器
    管理用户打断 AI 回复的逻辑
    """
    
    def __init__(
        self,
        min_transcribed_chars: int = 2,
        on_interrupt: Optional[callable] = None
    ):
        """
        初始化打断控制器
        
        Args:
            min_transcribed_chars: 触发打断的最小识别字符数
            on_interrupt: 打断回调
        """
        self.min_transcribed_chars = min_transcribed_chars
        self._on_interrupt = on_interrupt
        
        # 状态
        self._is_ai_speaking = False
        self._transcribed_text = ""
        self._lock = Lock()
        
        logger.info(f"BargeIn controller initialized: min_chars={min_transcribed_chars}")
    
    def set_ai_speaking(self, is_speaking: bool) -> None:
        """设置 AI 是否正在说话"""
        with self._lock:
            self._is_ai_speaking = is_speaking
            if not is_speaking:
                self._transcribed_text = ""
    
    def on_user_speech_detected(self, transcribed_text: str) -> bool:
        """
        用户语音检测到时调用
        
        Args:
            transcribed_text: 当前识别的文本
        
        Returns:
            是否触发打断
        """
        with self._lock:
            if not self._is_ai_speaking:
                return False
            
            self._transcribed_text = transcribed_text
            
            # 检查是否达到打断条件
            if len(transcribed_text.strip()) >= self.min_transcribed_chars:
                logger.info(f"Barge-in triggered: '{transcribed_text}'")
                
                if self._on_interrupt:
                    self._on_interrupt()
                
                return True
            
            return False
    
    @property
    def is_ai_speaking(self) -> bool:
        """AI 是否正在说话"""
        with self._lock:
            return self._is_ai_speaking


class VoiceAIClient:
    """
    实时语音对话客户端
    整合所有模块，提供完整的语音对话功能
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        use_gpu: bool = True
    ):
        """
        初始化客户端
        
        Args:
            server_url: 云端服务 URL
            use_gpu: 是否使用 GPU
        """
        self.server_url = server_url
        self.use_gpu = use_gpu
        
        # 状态
        self._state = ClientState.IDLE
        self._state_lock = Lock()
        self._is_running = Event()
        
        # 延迟指标
        self._metrics = LatencyMetrics()
        self._speech_start_time = 0.0
        
        # 初始化各模块
        self._init_modules()
        
        logger.info("VoiceAI Client initialized")
    
    def _init_modules(self) -> None:
        """初始化各模块"""
        # 全双工音频
        self.audio = FullDuplexAudio(
            capture_sample_rate=AudioConfig.SAMPLE_RATE,
            playback_sample_rate=TTSConfig.SAMPLE_RATE
        )
        
        # VAD 处理器
        self.vad = VADProcessor(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            use_gpu=self.use_gpu
        )
        
        # ASR 管理器
        self.asr = ASRManager(
            on_transcription=self._on_transcription,
            use_gpu=self.use_gpu
        )
        
        # 打断控制器
        self.barge_in = BargeInController(
            min_transcribed_chars=2,
            on_interrupt=self._do_interrupt
        )
        
        # WebSocket 客户端
        self.ws_client = WebSocketClient(
            server_url=self.server_url,
            on_text_message=self._on_text_message,
            on_audio_chunk=self._on_audio_chunk,
            on_connection_change=self._on_connection_change
        )
        
        # 音频接收器
        self.audio_receiver = AudioReceiver(
            on_audio_ready=self._on_audio_ready
        )
    
    def start(self) -> None:
        """启动客户端"""
        if self._is_running.is_set():
            logger.warning("Client already running")
            return
        
        self._is_running.set()
        
        # 启动 WebSocket
        self.ws_client.start()
        
        # 等待连接
        logger.info("Waiting for server connection...")
        if not self.ws_client.wait_connected(timeout=30.0):
            logger.error("Failed to connect to server")
            self._is_running.clear()
            raise RuntimeError("Server connection timeout")
        
        # 启动音频
        self.audio.start(self._on_audio_frame)
        
        self._set_state(ClientState.IDLE)
        logger.info("VoiceAI Client started - Ready for voice input")
    
    def stop(self) -> None:
        """停止客户端"""
        self._is_running.clear()
        
        self.audio.stop()
        self.ws_client.stop()
        
        logger.info("VoiceAI Client stopped")
    
    def _set_state(self, new_state: ClientState) -> None:
        """设置状态"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                logger.info(f"State: {old_state.value} -> {new_state.value}")
    
    @property
    def state(self) -> ClientState:
        """获取当前状态"""
        with self._state_lock:
            return self._state
    
    def _on_audio_frame(self, frame: np.ndarray) -> None:
        """
        音频帧回调
        
        Args:
            frame: 音频帧
        """
        # VAD 处理
        vad_result = self.vad.process_frame(frame)
        
        # 如果正在说话且 ASR 活动中，添加音频
        if vad_result.is_speech and self.asr.is_active:
            self.asr.add_audio(frame)
            
            # 打断检测（如果 AI 正在说话）
            if self.barge_in.is_ai_speaking:
                # 尝试获取部分识别结果
                buffer_duration = self.asr.streaming.get_buffer_duration_ms()
                if buffer_duration > 300:  # 300ms 后尝试识别
                    text = self.asr.streaming.process_accumulated(is_final=False)
                    if text:
                        self.barge_in.on_user_speech_detected(text)
    
    def _on_speech_start(self) -> None:
        """语音开始回调"""
        self._speech_start_time = time.time()
        
        # 获取预缓冲音频
        pre_buffer = self.audio.capture.get_pre_buffer()
        
        # 开始 ASR
        self.asr.start_recognition(pre_buffer)
        
        self._set_state(ClientState.LISTENING)
        logger.debug("Speech started - ASR activated")
    
    def _on_speech_end(self) -> None:
        """语音结束回调"""
        # 完成 ASR
        final_text = self.asr.end_recognition()
        
        if final_text and final_text.strip():
            # 发送到云端
            self.ws_client.send_asr_result(final_text, is_final=True)
            self._set_state(ClientState.PROCESSING)
            
            # 记录延迟
            self._metrics.asr_latency_ms = (time.time() - self._speech_start_time) * 1000
            logger.info(f"Speech ended: '{final_text}' (ASR latency: {self._metrics.asr_latency_ms:.1f}ms)")
        else:
            # 空识别结果，返回空闲
            self._set_state(ClientState.IDLE)
            logger.debug("Speech ended with no text")
    
    def _on_transcription(self, text: str, is_final: bool) -> None:
        """
        转写结果回调
        
        Args:
            text: 识别文本
            is_final: 是否为最终结果
        """
        if not is_final:
            # 发送部分结果
            self.ws_client.send_asr_result(text, is_final=False)
    
    def _on_text_message(self, data: Dict[str, Any]) -> None:
        """
        文本消息回调
        
        Args:
            data: 消息数据
        """
        msg_type = data.get('type', '')
        content = data.get('content', '')
        
        if msg_type == MessageType.TTS_START.value:
            # TTS 开始
            self._set_state(ClientState.PLAYING)
            self.barge_in.set_ai_speaking(True)
            self.audio_receiver.reset()
            logger.debug("TTS playback starting")
            
        elif msg_type == MessageType.TTS_END.value:
            # TTS 结束
            self.barge_in.set_ai_speaking(False)
            # 等待播放完成后设置为 IDLE
            logger.debug("TTS stream ended")
            
        elif msg_type == MessageType.LLM_TOKEN.value:
            # LLM token 流（可用于显示）
            pass
            
        elif msg_type == MessageType.LLM_END.value:
            # LLM 生成结束
            logger.debug("LLM generation completed")
            
        elif msg_type == MessageType.INTERRUPT_ACK.value:
            # 打断确认
            self._set_state(ClientState.LISTENING)
            logger.debug("Interrupt acknowledged")
            
        elif msg_type == MessageType.ERROR.value:
            # 错误
            logger.error(f"Server error: {content}")
            self._set_state(ClientState.IDLE)
    
    def _on_audio_chunk(self, header: AudioChunkHeader, pcm_data: bytes) -> None:
        """
        音频块回调
        
        Args:
            header: 音频块头部
            pcm_data: PCM 数据
        """
        self.audio_receiver.on_audio_chunk(header, pcm_data)
    
    def _on_audio_ready(self, audio: np.ndarray) -> None:
        """
        音频就绪回调
        
        Args:
            audio: 音频数据
        """
        # 播放音频
        self.audio.play_audio(audio)
        
        # 检查是否播放完成
        if not self.audio_receiver.is_receiving and not self.audio.is_playing():
            self.barge_in.set_ai_speaking(False)
            if self.state == ClientState.PLAYING:
                self._set_state(ClientState.IDLE)
    
    def _on_connection_change(self, is_connected: bool) -> None:
        """
        连接状态变化回调
        
        Args:
            is_connected: 是否已连接
        """
        if is_connected:
            logger.info("Connected to server")
        else:
            logger.warning("Disconnected from server")
            if self._is_running.is_set():
                self._set_state(ClientState.IDLE)
    
    def _do_interrupt(self) -> None:
        """执行打断"""
        self._set_state(ClientState.INTERRUPTING)
        
        # 停止播放
        self.audio.stop_playback()
        
        # 重置音频接收器
        self.audio_receiver.reset()
        
        # 发送打断信号
        self.ws_client.send_interrupt()
        
        logger.info("Interrupt executed")
    
    def get_metrics(self) -> LatencyMetrics:
        """获取延迟指标"""
        self._metrics.network_rtt_ms = self.ws_client.get_status().last_rtt_ms
        self._metrics.timestamp = time.time()
        return self._metrics
    
    def run_interactive(self) -> None:
        """运行交互式会话"""
        print("\n" + "="*50)
        print("  VoiceAI 实时语音对话系统 - 本地客户端")
        print("="*50)
        print("\n说话开始对话，按 Ctrl+C 退出\n")
        
        try:
            self.start()
            
            while self._is_running.is_set():
                time.sleep(0.1)
                
                # 显示状态
                state = self.state
                metrics = self.get_metrics()
                
                status_line = f"\r状态: {state.value:<15} | RTT: {metrics.network_rtt_ms:.0f}ms"
                print(status_line, end='', flush=True)
                
        except KeyboardInterrupt:
            print("\n\n正在关闭...")
        finally:
            self.stop()
            print("已退出")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VoiceAI Local Client')
    parser.add_argument('--server', '-s', default='ws://localhost:8765',
                       help='Server WebSocket URL')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行客户端
    client = VoiceAIClient(
        server_url=args.server,
        use_gpu=not args.no_gpu
    )
    
    client.run_interactive()


if __name__ == "__main__":
    main()
