"""
TTS 引擎模块
基于 GPT-SoVITS v4 实现高保真语音合成
"""

import numpy as np
from typing import Optional, Callable, AsyncGenerator, Generator
from dataclasses import dataclass
import asyncio
from queue import Queue, Empty
from threading import Thread, Event, Lock
import logging
import time
import io

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS 配置"""
    model_path: str = ""
    ref_audio_path: str = ""
    ref_text: str = ""
    sample_rate: int = 48000
    streaming_mode: int = 3  # GPT-SoVITS streaming mode
    overlap_length: int = 2  # 重叠长度，防止块边界伪影
    speed: float = 1.0
    language: str = "zh"


@dataclass  
class AudioChunk:
    """音频块"""
    pcm_data: bytes
    sample_rate: int
    sequence: int
    is_last: bool
    text: str = ""


class GPTSoVITSEngine:
    """
    GPT-SoVITS TTS 引擎
    支持流式合成
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        初始化 TTS 引擎
        
        Args:
            config: TTS 配置
        """
        self.config = config or TTSConfig()
        self._model = None
        self._is_synthesizing = False
        self._should_stop = False
        self._lock = Lock()
        
        logger.info("GPT-SoVITS Engine initializing")
    
    def load_model(self) -> None:
        """加载模型"""
        try:
            # 尝试导入 GPT-SoVITS
            # 注意：实际使用时需要正确安装 GPT-SoVITS
            logger.info("Loading GPT-SoVITS model...")
            
            # GPT-SoVITS 的导入和初始化
            # 这里需要根据实际的 GPT-SoVITS API 进行调整
            try:
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
                
                tts_config = TTS_Config("")
                tts_config.device = "cuda"
                tts_config.is_half = True
                
                self._model = TTS(tts_config)
                
                # 加载参考音频
                if self.config.ref_audio_path:
                    logger.info(f"Loading reference audio: {self.config.ref_audio_path}")
                
                logger.info("GPT-SoVITS model loaded")
                
            except ImportError:
                logger.warning("GPT-SoVITS not installed, using mock TTS")
                self._model = None
                
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise RuntimeError(f"TTS model loading failed: {e}")
    
    async def synthesize_stream(
        self,
        text: str,
        on_chunk: Optional[Callable[[AudioChunk], None]] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        流式合成语音
        
        Args:
            text: 要合成的文本
            on_chunk: 音频块回调
        
        Yields:
            AudioChunk: 音频块
        """
        with self._lock:
            self._is_synthesizing = True
            self._should_stop = False
        
        sequence = 0
        
        try:
            if self._model is None:
                # 使用模拟合成
                async for chunk in self._mock_synthesize(text):
                    if self._should_stop:
                        break
                    
                    chunk.sequence = sequence
                    sequence += 1
                    
                    if on_chunk:
                        on_chunk(chunk)
                    
                    yield chunk
            else:
                # 使用真实模型
                async for chunk in self._real_synthesize(text):
                    if self._should_stop:
                        break
                    
                    chunk.sequence = sequence
                    sequence += 1
                    
                    if on_chunk:
                        on_chunk(chunk)
                    
                    yield chunk
                    
        finally:
            with self._lock:
                self._is_synthesizing = False
    
    async def _real_synthesize(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """使用真实模型合成"""
        try:
            # GPT-SoVITS streaming synthesis
            # 注意：需要根据实际 API 调整
            
            inputs = {
                "text": text,
                "text_lang": self.config.language,
                "ref_audio_path": self.config.ref_audio_path,
                "prompt_text": self.config.ref_text,
                "prompt_lang": self.config.language,
                "top_k": 5,
                "top_p": 1,
                "temperature": 1,
                "speed_factor": self.config.speed,
                "streaming_mode": True,
                "return_fragment": True,
            }
            
            # 调用模型进行流式合成
            for chunk_data in self._model.run(inputs):
                if self._should_stop:
                    break
                
                # 转换采样率如果需要
                audio_data = chunk_data
                
                # 转换为 int16 PCM
                if isinstance(audio_data, np.ndarray):
                    if audio_data.dtype == np.float32:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    pcm_bytes = audio_data.tobytes()
                else:
                    pcm_bytes = audio_data
                
                yield AudioChunk(
                    pcm_data=pcm_bytes,
                    sample_rate=self.config.sample_rate,
                    sequence=0,
                    is_last=False,
                    text=text
                )
                
                await asyncio.sleep(0)
            
            # 最后一个块
            yield AudioChunk(
                pcm_data=b"",
                sample_rate=self.config.sample_rate,
                sequence=0,
                is_last=True,
                text=text
            )
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            yield AudioChunk(
                pcm_data=b"",
                sample_rate=self.config.sample_rate,
                sequence=0,
                is_last=True,
                text=text
            )
    
    async def _mock_synthesize(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """模拟合成（用于测试）"""
        logger.debug(f"Mock synthesizing: '{text}'")
        
        # 生成模拟音频（正弦波）
        duration = len(text) * 0.15  # 每个字符约 150ms
        duration = max(0.5, min(duration, 10.0))  # 限制在 0.5-10 秒
        
        sample_rate = self.config.sample_rate
        total_samples = int(duration * sample_rate)
        chunk_size = int(0.1 * sample_rate)  # 100ms 每块
        
        # 生成正弦波
        t = np.linspace(0, duration, total_samples, dtype=np.float32)
        frequency = 440  # A4 音符
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # 添加一些变化
        envelope = np.exp(-t / duration)
        audio = audio * envelope
        
        # 转换为 int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 分块输出
        offset = 0
        while offset < len(audio_int16):
            if self._should_stop:
                break
            
            end = min(offset + chunk_size, len(audio_int16))
            chunk_data = audio_int16[offset:end]
            
            yield AudioChunk(
                pcm_data=chunk_data.tobytes(),
                sample_rate=sample_rate,
                sequence=0,
                is_last=(end >= len(audio_int16)),
                text=text
            )
            
            offset = end
            await asyncio.sleep(0.05)  # 模拟合成延迟
    
    def interrupt(self) -> None:
        """中断当前合成"""
        with self._lock:
            if self._is_synthesizing:
                self._should_stop = True
                logger.info("TTS synthesis interrupted")
    
    @property
    def is_synthesizing(self) -> bool:
        """是否正在合成"""
        with self._lock:
            return self._is_synthesizing


class TTSQueue:
    """
    TTS 队列
    管理多个文本块的合成
    """
    
    def __init__(
        self,
        tts_engine: GPTSoVITSEngine,
        on_audio_ready: Optional[Callable[[AudioChunk], None]] = None
    ):
        """
        初始化 TTS 队列
        
        Args:
            tts_engine: TTS 引擎
            on_audio_ready: 音频就绪回调
        """
        self.tts = tts_engine
        self._on_audio_ready = on_audio_ready
        
        # 文本队列
        self._text_queue: Queue = Queue()
        
        # 状态
        self._is_running = Event()
        self._process_thread: Optional[Thread] = None
        
        # 序列号
        self._global_sequence = 0
        self._sequence_lock = Lock()
    
    def start(self) -> None:
        """启动队列处理"""
        if self._is_running.is_set():
            return
        
        self._is_running.set()
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        logger.info("TTS queue started")
    
    def stop(self) -> None:
        """停止队列处理"""
        self._is_running.clear()
        self._text_queue.put(None)  # 发送停止信号
        
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
        
        logger.info("TTS queue stopped")
    
    def enqueue(self, text: str, sequence: int) -> None:
        """
        添加文本到队列
        
        Args:
            text: 要合成的文本
            sequence: 序列号
        """
        self._text_queue.put((text, sequence))
    
    def clear(self) -> None:
        """清空队列"""
        # 清空队列
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except Empty:
                break
        
        # 中断当前合成
        self.tts.interrupt()
        
        logger.info("TTS queue cleared")
    
    def _process_loop(self) -> None:
        """处理循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._is_running.is_set():
                try:
                    item = self._text_queue.get(timeout=0.1)
                    
                    if item is None:
                        break
                    
                    text, text_sequence = item
                    
                    # 异步合成
                    loop.run_until_complete(self._synthesize(text, text_sequence))
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"TTS processing error: {e}")
        finally:
            loop.close()
    
    async def _synthesize(self, text: str, text_sequence: int) -> None:
        """执行合成"""
        async for chunk in self.tts.synthesize_stream(text):
            if not self._is_running.is_set():
                break
            
            # 更新全局序列号
            with self._sequence_lock:
                chunk.sequence = self._global_sequence
                self._global_sequence += 1
            
            if self._on_audio_ready:
                self._on_audio_ready(chunk)


class MockTTSEngine:
    """
    模拟 TTS 引擎
    用于测试
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._is_synthesizing = False
        self._should_stop = False
        logger.info("MockTTSEngine initialized")
    
    def load_model(self) -> None:
        logger.info("Mock TTS model loaded")
    
    async def synthesize_stream(
        self,
        text: str,
        on_chunk: Optional[Callable[[AudioChunk], None]] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """模拟流式合成"""
        self._is_synthesizing = True
        self._should_stop = False
        
        try:
            # 每个字符生成约 100ms 音频
            duration_per_char = 0.1
            total_duration = len(text) * duration_per_char
            total_duration = max(0.3, min(total_duration, 10.0))
            
            sample_rate = self.config.sample_rate
            chunk_duration = 0.1  # 100ms per chunk
            chunk_samples = int(chunk_duration * sample_rate)
            
            num_chunks = max(1, int(total_duration / chunk_duration))
            
            for i in range(num_chunks):
                if self._should_stop:
                    break
                
                # 生成静音块（实际应用中会是合成的语音）
                audio_data = np.zeros(chunk_samples, dtype=np.int16)
                
                # 添加一点噪声使其不完全静音
                audio_data = (np.random.randn(chunk_samples) * 100).astype(np.int16)
                
                is_last = (i == num_chunks - 1)
                
                chunk = AudioChunk(
                    pcm_data=audio_data.tobytes(),
                    sample_rate=sample_rate,
                    sequence=i,
                    is_last=is_last,
                    text=text if is_last else ""
                )
                
                if on_chunk:
                    on_chunk(chunk)
                
                yield chunk
                
                await asyncio.sleep(0.02)  # 模拟合成时间
                
        finally:
            self._is_synthesizing = False
    
    def interrupt(self) -> None:
        self._should_stop = True
    
    @property
    def is_synthesizing(self) -> bool:
        return self._is_synthesizing


def create_tts_engine(use_mock: bool = False, config: Optional[TTSConfig] = None):
    """
    创建 TTS 引擎
    
    Args:
        use_mock: 是否使用模拟引擎
        config: TTS 配置
    
    Returns:
        TTS 引擎实例
    """
    if use_mock:
        return MockTTSEngine(config)
    return GPTSoVITSEngine(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_tts():
        """测试 TTS"""
        tts = MockTTSEngine()
        tts.load_model()
        
        print("Testing TTS synthesis:")
        
        chunks = []
        async for chunk in tts.synthesize_stream("你好，这是一个测试。"):
            chunks.append(chunk)
            print(f"Chunk #{chunk.sequence}: {len(chunk.pcm_data)} bytes, is_last={chunk.is_last}")
        
        print(f"\nTotal chunks: {len(chunks)}")
    
    asyncio.run(test_tts())
