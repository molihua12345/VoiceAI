"""
音频采集与管理模块
实现音频采集、环形缓冲区和全双工音频处理
"""

import numpy as np
import sounddevice as sd
from collections import deque
from threading import Lock, Event
from typing import Callable, Optional
import logging
import time

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import AudioConfig

logger = logging.getLogger(__name__)


class RingBuffer:
    """线程安全的环形缓冲区"""
    
    def __init__(self, max_frames: int):
        """
        初始化环形缓冲区
        
        Args:
            max_frames: 最大帧数
        """
        self.buffer: deque = deque(maxlen=max_frames)
        self.lock = Lock()
    
    def push(self, frame: np.ndarray) -> None:
        """添加一帧到缓冲区"""
        with self.lock:
            self.buffer.append(frame.copy())
    
    def get_all(self) -> np.ndarray:
        """获取缓冲区所有数据"""
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self.buffer))
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


class AudioCapture:
    """
    音频采集模块
    使用非阻塞回调模式实现实时音频采集
    """
    
    def __init__(
        self,
        sample_rate: int = AudioConfig.SAMPLE_RATE,
        channels: int = AudioConfig.CHANNELS,
        frame_size: int = AudioConfig.FRAME_SIZE,
        pre_buffer_frames: int = AudioConfig.PRE_BUFFER_FRAMES,
        device: Optional[int] = None
    ):
        """
        初始化音频采集
        
        Args:
            sample_rate: 采样率
            channels: 通道数
            frame_size: 每帧样本数
            pre_buffer_frames: 预缓冲帧数 (用于保留语音起始)
            device: 音频设备索引，None 为默认设备
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.device = device
        
        # 环形预缓冲区 - 保存最近的音频帧
        self.pre_buffer = RingBuffer(max_frames=pre_buffer_frames)
        
        # 回调函数
        self._on_audio_frame: Optional[Callable[[np.ndarray], None]] = None
        
        # 状态
        self._stream: Optional[sd.InputStream] = None
        self._is_running = Event()
        
        # 统计
        self._frame_count = 0
        self._overflow_count = 0
        
        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {channels}ch, {frame_size} samples/frame")
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """
        音频采集回调函数 (在独立线程中运行)
        
        Args:
            indata: 输入音频数据
            frames: 帧数
            time_info: 时间信息
            status: 状态标志
        """
        if status.input_overflow:
            self._overflow_count += 1
            logger.warning(f"Input overflow detected (count: {self._overflow_count})")
        
        # 转换为 float32 单通道
        audio_frame = indata[:, 0].astype(np.float32)
        
        # 保存到预缓冲区
        self.pre_buffer.push(audio_frame)
        
        # 调用用户回调
        if self._on_audio_frame is not None:
            try:
                self._on_audio_frame(audio_frame)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        
        self._frame_count += 1
    
    def start(self, on_audio_frame: Callable[[np.ndarray], None]) -> None:
        """
        开始音频采集
        
        Args:
            on_audio_frame: 音频帧回调函数
        """
        if self._is_running.is_set():
            logger.warning("Audio capture already running")
            return
        
        self._on_audio_frame = on_audio_frame
        self._frame_count = 0
        self._overflow_count = 0
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.frame_size,
                device=self.device,
                callback=self._audio_callback,
                latency='low'  # 低延迟模式
            )
            self._stream.start()
            self._is_running.set()
            logger.info("Audio capture started")
            
        except sd.PortAudioError as e:
            logger.error(f"Failed to start audio capture: {e}")
            logger.error("Please check WASAPI/ASIO configuration")
            raise RuntimeError(f"Audio driver initialization failed: {e}")
    
    def stop(self) -> None:
        """停止音频采集"""
        if not self._is_running.is_set():
            return
        
        self._is_running.clear()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        logger.info(f"Audio capture stopped. Frames: {self._frame_count}, Overflows: {self._overflow_count}")
    
    def get_pre_buffer(self) -> np.ndarray:
        """获取预缓冲区数据 (用于获取语音起始)"""
        return self.pre_buffer.get_all()
    
    def clear_pre_buffer(self) -> None:
        """清空预缓冲区"""
        self.pre_buffer.clear()
    
    @property
    def is_running(self) -> bool:
        """是否正在采集"""
        return self._is_running.is_set()
    
    @staticmethod
    def list_devices() -> list:
        """列出可用的音频设备"""
        return sd.query_devices()
    
    @staticmethod
    def get_default_device() -> tuple:
        """获取默认输入/输出设备"""
        return sd.default.device


class AudioPlayer:
    """
    音频播放模块
    使用非阻塞回调模式实现实时音频播放
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,  # TTS 输出采样率
        channels: int = 1,
        buffer_duration_ms: int = 200,  # 缓冲区时长
        device: Optional[int] = None
    ):
        """
        初始化音频播放
        
        Args:
            sample_rate: 采样率
            channels: 通道数
            buffer_duration_ms: 缓冲区时长（毫秒）
            device: 音频设备索引
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        
        # 播放缓冲区
        self._buffer: deque = deque()
        self._buffer_lock = Lock()
        
        # 缓冲区大小阈值
        buffer_samples = int(sample_rate * buffer_duration_ms / 1000)
        self._min_buffer_samples = buffer_samples // 2
        
        # 状态
        self._stream: Optional[sd.OutputStream] = None
        self._is_running = Event()
        self._is_playing = Event()
        
        # 统计
        self._underrun_count = 0
        
        # 回调
        self._on_playback_complete: Optional[Callable[[], None]] = None
        
        logger.info(f"AudioPlayer initialized: {sample_rate}Hz, {channels}ch")
    
    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """
        音频播放回调函数
        """
        if status.output_underflow:
            self._underrun_count += 1
            logger.warning(f"Output underrun detected (count: {self._underrun_count})")
        
        with self._buffer_lock:
            if len(self._buffer) >= frames:
                # 有足够数据
                for i in range(frames):
                    outdata[i, 0] = self._buffer.popleft()
                self._is_playing.set()
            else:
                # 数据不足，填充静音
                available = len(self._buffer)
                for i in range(available):
                    outdata[i, 0] = self._buffer.popleft()
                outdata[available:, 0] = 0
                
                if self._is_playing.is_set() and available == 0:
                    self._is_playing.clear()
                    if self._on_playback_complete:
                        self._on_playback_complete()
    
    def start(self) -> None:
        """开始播放流"""
        if self._is_running.is_set():
            return
        
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=1024,
                device=self.device,
                callback=self._audio_callback,
                latency='low'
            )
            self._stream.start()
            self._is_running.set()
            logger.info("Audio player started")
            
        except sd.PortAudioError as e:
            logger.error(f"Failed to start audio player: {e}")
            raise RuntimeError(f"Audio player initialization failed: {e}")
    
    def stop(self) -> None:
        """停止播放流"""
        if not self._is_running.is_set():
            return
        
        self._is_running.clear()
        self._is_playing.clear()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        logger.info(f"Audio player stopped. Underruns: {self._underrun_count}")
    
    def enqueue(self, audio_data: np.ndarray) -> None:
        """
        添加音频数据到播放队列
        
        Args:
            audio_data: float32 格式的音频数据
        """
        with self._buffer_lock:
            self._buffer.extend(audio_data.flatten())
    
    def clear_buffer(self) -> None:
        """清空播放缓冲区"""
        with self._buffer_lock:
            self._buffer.clear()
        self._is_playing.clear()
        logger.info("Playback buffer cleared")
    
    def get_buffer_level(self) -> int:
        """获取缓冲区中的样本数"""
        with self._buffer_lock:
            return len(self._buffer)
    
    @property
    def is_running(self) -> bool:
        """播放流是否运行中"""
        return self._is_running.is_set()
    
    @property
    def is_playing(self) -> bool:
        """是否正在播放音频"""
        return self._is_playing.is_set()
    
    def set_playback_complete_callback(self, callback: Callable[[], None]) -> None:
        """设置播放完成回调"""
        self._on_playback_complete = callback


class FullDuplexAudio:
    """
    全双工音频管理器
    同时管理音频采集和播放，支持打断检测
    """
    
    def __init__(
        self,
        capture_sample_rate: int = AudioConfig.SAMPLE_RATE,
        playback_sample_rate: int = 48000
    ):
        self.capture = AudioCapture(sample_rate=capture_sample_rate)
        self.player = AudioPlayer(sample_rate=playback_sample_rate)
    
    def start(self, on_audio_frame: Callable[[np.ndarray], None]) -> None:
        """启动全双工音频"""
        self.player.start()
        self.capture.start(on_audio_frame)
    
    def stop(self) -> None:
        """停止全双工音频"""
        self.capture.stop()
        self.player.stop()
    
    def is_playing(self) -> bool:
        """是否正在播放"""
        return self.player.is_playing
    
    def play_audio(self, audio_data: np.ndarray) -> None:
        """播放音频"""
        self.player.enqueue(audio_data)
    
    def stop_playback(self) -> None:
        """停止播放并清空缓冲区"""
        self.player.clear_buffer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Available audio devices:")
    print(AudioCapture.list_devices())
    
    print(f"\nDefault devices: {AudioCapture.get_default_device()}")
    
    # 简单录音测试
    frames = []
    
    def on_frame(frame):
        frames.append(frame)
    
    capture = AudioCapture()
    capture.start(on_frame)
    
    print("\nRecording for 3 seconds...")
    time.sleep(3)
    
    capture.stop()
    
    audio = np.concatenate(frames) if frames else np.array([])
    print(f"Recorded {len(audio)} samples ({len(audio)/AudioConfig.SAMPLE_RATE:.2f} seconds)")
