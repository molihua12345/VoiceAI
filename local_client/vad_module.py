"""
语音活动检测模块 (VAD)
基于 Silero-VAD 实现，支持实时语音端点检测
"""

import numpy as np
import torch
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from threading import Lock
import logging
import time

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    """VAD 检测结果"""
    is_speech: bool           # 当前帧是否为语音
    is_speech_start: bool     # 是否为语音起始
    is_speech_end: bool       # 是否为语音结束
    confidence: float         # 置信度
    timestamp: float          # 时间戳


class SileroVAD:
    """
    基于 Silero-VAD 的语音活动检测
    """
    
    def __init__(
        self,
        sample_rate: int = AudioConfig.SAMPLE_RATE,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        use_gpu: bool = True
    ):
        """
        初始化 VAD 模块
        
        Args:
            sample_rate: 采样率 (必须是 8000 或 16000)
            threshold: 语音检测阈值
            min_speech_duration_ms: 最小语音持续时间
            min_silence_duration_ms: 最小静音持续时间（用于判断语音结束）
            speech_pad_ms: 语音前后填充
            use_gpu: 是否使用 GPU
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # 设备选择
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("VAD using GPU acceleration")
        else:
            self.device = torch.device('cpu')
            logger.info("VAD using CPU")
        
        # 加载 Silero VAD 模型
        self._load_model()
        
        # 状态
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._current_speech_duration = 0.0
        
        # 帧计数
        self._frame_duration_ms = AudioConfig.FRAME_DURATION_MS
        self._frames_since_speech = 0
        self._frames_with_speech = 0
        
        # 线程安全
        self._lock = Lock()
        
        logger.info(f"SileroVAD initialized: threshold={threshold}, min_speech={min_speech_duration_ms}ms")
    
    def _load_model(self) -> None:
        """加载 Silero VAD 模型"""
        import os
        
        # 设置纯英文路径避免中文路径问题
        self._setup_torch_home()
        
        # 尝试多种加载方式
        load_methods = [
            self._load_from_hub,
            self._load_from_local_cache,
        ]
        
        last_error = None
        for method in load_methods:
            try:
                method()
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Load method {method.__name__} failed: {e}")
                continue
        
        # 所有方法都失败了
        logger.error(f"Failed to load Silero VAD model: {last_error}")
        raise RuntimeError(
            f"VAD model loading failed: {last_error}\n"
            "解决方案:\n"
            "1. 设置 TORCH_HOME 到纯英文路径: $env:TORCH_HOME='C:/torch_cache'\n"
            "2. 手动下载模型到 C:/torch_cache/hub/snakers4_silero-vad_master/\n"
            "3. 检查网络连接"
        )
    
    def _setup_torch_home(self) -> None:
        """设置 TORCH_HOME 到纯英文路径，避免中文路径问题"""
        import os
        
        # 检查当前 TORCH_HOME 是否包含非 ASCII 字符
        current_home = os.environ.get('TORCH_HOME', '')
        user_home = os.path.expanduser('~')
        
        def has_non_ascii(path: str) -> bool:
            try:
                path.encode('ascii')
                return False
            except UnicodeEncodeError:
                return True
        
        # 如果已设置且是纯 ASCII 路径，直接返回
        if current_home and not has_non_ascii(current_home):
            return
        
        # 如果用户目录包含中文，设置到纯英文路径
        if has_non_ascii(user_home):
            # 使用项目目录下的 .cache 或系统盘根目录
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 优先使用项目目录（如果是纯 ASCII）
            if not has_non_ascii(project_dir):
                torch_home = os.path.join(project_dir, '.torch_cache')
            else:
                # 否则使用 C:/torch_cache
                torch_home = 'C:/torch_cache'
            
            os.makedirs(torch_home, exist_ok=True)
            os.environ['TORCH_HOME'] = torch_home
            logger.info(f"Set TORCH_HOME to ASCII path: {torch_home}")
    
    def _load_from_hub(self) -> None:
        """从 torch.hub 加载模型"""
        import os
        
        # 设置信任缓存，避免每次都验证
        os.environ.setdefault('TORCH_HUB_SKIP_VERIFY', '1')
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        
        self._setup_model(model, utils)
        logger.info("Silero VAD model loaded from torch.hub")
    
    def _load_from_local_cache(self) -> None:
        """从本地缓存加载模型"""
        import os
        import sys
        
        # 常见的缓存路径
        cache_paths = [
            os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master'),
            os.path.expanduser('~/torch/hub/snakers4_silero-vad_master'),
            os.path.join(os.environ.get('TORCH_HOME', ''), 'hub/snakers4_silero-vad_master'),
        ]
        
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                logger.info(f"Found local cache at: {cache_path}")
                
                # 添加到 sys.path
                if cache_path not in sys.path:
                    sys.path.insert(0, cache_path)
                
                model, utils = torch.hub.load(
                    repo_or_dir=cache_path,
                    model='silero_vad',
                    source='local',
                    onnx=False,
                    trust_repo=True
                )
                
                self._setup_model(model, utils)
                logger.info("Silero VAD model loaded from local cache")
                return
        
        raise FileNotFoundError("No local cache found")
    
    def _setup_model(self, model, utils) -> None:
        """设置模型"""
        self.model = model.to(self.device)
        self.model.eval()
        
        # 获取工具函数
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        
        self.get_speech_timestamps = get_speech_timestamps
        
        # 重置模型状态
        self.model.reset_states()
    
    def reset(self) -> None:
        """重置 VAD 状态"""
        with self._lock:
            self._is_speaking = False
            self._speech_start_time = None
            self._silence_start_time = None
            self._current_speech_duration = 0.0
            self._frames_since_speech = 0
            self._frames_with_speech = 0
            self.model.reset_states()
            logger.debug("VAD state reset")
    
    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """
        处理单帧音频
        
        Args:
            audio_frame: float32 格式的音频帧 (32ms)
        
        Returns:
            VADResult: 检测结果
        """
        current_time = time.time()
        
        with self._lock:
            # 转换为 tensor
            audio_tensor = torch.from_numpy(audio_frame).float().to(self.device)
            
            # 获取语音概率
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            is_speech = speech_prob >= self.threshold
            is_speech_start = False
            is_speech_end = False
            
            if is_speech:
                self._frames_with_speech += 1
                self._frames_since_speech = 0
                
                if not self._is_speaking:
                    # 检查是否达到最小语音持续时间
                    speech_duration = self._frames_with_speech * self._frame_duration_ms
                    if speech_duration >= self.min_speech_duration_ms:
                        self._is_speaking = True
                        self._speech_start_time = current_time
                        self._silence_start_time = None
                        is_speech_start = True
                        logger.debug(f"Speech start detected (confidence: {speech_prob:.2f})")
                else:
                    self._silence_start_time = None
            else:
                self._frames_since_speech += 1
                
                if self._is_speaking:
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                    
                    # 检查静音持续时间
                    silence_duration = self._frames_since_speech * self._frame_duration_ms
                    if silence_duration >= self.min_silence_duration_ms:
                        self._is_speaking = False
                        self._speech_start_time = None
                        self._frames_with_speech = 0
                        is_speech_end = True
                        logger.debug(f"Speech end detected after {silence_duration}ms silence")
                else:
                    # 重置语音帧计数
                    if self._frames_since_speech > 10:
                        self._frames_with_speech = 0
            
            return VADResult(
                is_speech=self._is_speaking,
                is_speech_start=is_speech_start,
                is_speech_end=is_speech_end,
                confidence=speech_prob,
                timestamp=current_time
            )
    
    @property
    def is_speaking(self) -> bool:
        """是否正在说话"""
        with self._lock:
            return self._is_speaking


class VADProcessor:
    """
    VAD 处理器
    集成 VAD 检测和回调管理
    """
    
    def __init__(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_speech_frame: Optional[Callable[[np.ndarray, float], None]] = None,
        **vad_kwargs
    ):
        """
        初始化 VAD 处理器
        
        Args:
            on_speech_start: 语音开始回调
            on_speech_end: 语音结束回调
            on_speech_frame: 语音帧回调 (frame, confidence)
            **vad_kwargs: VAD 参数
        """
        self.vad = SileroVAD(**vad_kwargs)
        
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_speech_frame = on_speech_frame
        
        # 累积的语音帧
        self._speech_frames: list = []
        self._lock = Lock()
    
    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """
        处理音频帧
        
        Args:
            audio_frame: 音频帧
        
        Returns:
            VADResult: 检测结果
        """
        result = self.vad.process_frame(audio_frame)
        
        if result.is_speech_start:
            with self._lock:
                self._speech_frames = []
            if self._on_speech_start:
                self._on_speech_start()
        
        if result.is_speech:
            with self._lock:
                self._speech_frames.append(audio_frame.copy())
            if self._on_speech_frame:
                self._on_speech_frame(audio_frame, result.confidence)
        
        if result.is_speech_end:
            if self._on_speech_end:
                self._on_speech_end()
        
        return result
    
    def get_speech_audio(self) -> np.ndarray:
        """获取累积的语音音频"""
        with self._lock:
            if not self._speech_frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._speech_frames)
    
    def clear_speech_buffer(self) -> None:
        """清空语音缓冲"""
        with self._lock:
            self._speech_frames = []
    
    def reset(self) -> None:
        """重置处理器"""
        self.vad.reset()
        self.clear_speech_buffer()
    
    @property
    def is_speaking(self) -> bool:
        """是否正在说话"""
        return self.vad.is_speaking


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建 VAD
    vad = SileroVAD()
    
    # 生成测试音频 (静音 + 噪声)
    silence = np.zeros(512, dtype=np.float32)
    noise = np.random.randn(512).astype(np.float32) * 0.1
    
    print("Testing with silence:")
    result = vad.process_frame(silence)
    print(f"  is_speech: {result.is_speech}, confidence: {result.confidence:.3f}")
    
    print("\nTesting with noise:")
    result = vad.process_frame(noise)
    print(f"  is_speech: {result.is_speech}, confidence: {result.confidence:.3f}")
    
    print("\nVAD module test complete")
