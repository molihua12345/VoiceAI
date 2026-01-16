"""
TTS 引擎模块
基于 GPT-SoVITS v4 实现高保真语音合成
通过 HTTP API 调用 GPT-SoVITS 服务
"""

import numpy as np
from typing import Optional, Callable, AsyncGenerator, Generator, Union
from dataclasses import dataclass, field
import asyncio
from queue import Queue, Empty
from threading import Thread, Event, Lock
import logging
import time
import io
import aiohttp
import struct

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS 配置"""
    # GPT-SoVITS API 服务地址
    api_url: str = "http://127.0.0.1:9880"
    
    # 参考音频配置
    ref_audio_path: str = ""  # 参考音频路径（服务端可访问的路径）
    prompt_text: str = ""     # 参考音频对应的文本
    prompt_lang: str = "zh"   # 参考音频语言
    
    # 合成参数
    sample_rate: int = 32000  # GPT-SoVITS v4 输出采样率
    text_lang: str = "zh"     # 合成文本语言
    
    # 流式模式: 0=关闭, 1=最佳质量, 2=中等质量, 3=快速响应
    streaming_mode: int = 3
    
    # 推理参数
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed_factor: float = 1.0
    repetition_penalty: float = 1.35
    
    # 流式参数
    overlap_length: int = 2      # 语义token重叠长度
    min_chunk_length: int = 16   # 最小块长度
    fragment_interval: float = 0.3  # 分段间隔
    
    # 文本切分方法
    text_split_method: str = "cut5"
    
    # 批处理
    batch_size: int = 1
    parallel_infer: bool = True


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
    通过 HTTP API 调用 GPT-SoVITS 服务进行流式语音合成
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        初始化 TTS 引擎
        
        Args:
            config: TTS 配置
        """
        self.config = config or TTSConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_synthesizing = False
        self._should_stop = False
        self._lock = Lock()
        self._is_ready = False
        
        logger.info(f"GPT-SoVITS Engine initializing, API URL: {self.config.api_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def load_model(self) -> None:
        """
        检查 GPT-SoVITS API 服务是否可用
        注意：实际模型加载由 GPT-SoVITS 服务端完成
        """
        import urllib.request
        import urllib.error
        
        try:
            # 简单检查服务是否在线
            url = f"{self.config.api_url}/set_refer_audio"
            req = urllib.request.Request(url, method='GET')
            
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    # 即使返回400也说明服务在线
                    pass
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    # 400 错误表示服务在线但参数缺失，这是正常的
                    pass
                else:
                    raise
            
            self._is_ready = True
            logger.info(f"GPT-SoVITS API service is ready at {self.config.api_url}")
            
        except urllib.error.URLError as e:
            logger.warning(f"GPT-SoVITS API service not available at {self.config.api_url}: {e}")
            logger.warning("TTS will use mock mode. Start GPT-SoVITS API server with:")
            logger.warning(f"  python api_v2.py -a 127.0.0.1 -p 9880")
            self._is_ready = False
        except Exception as e:
            logger.warning(f"Failed to connect to GPT-SoVITS API: {e}")
            self._is_ready = False
    
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
            if not self._is_ready:
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
                # 使用 HTTP API 调用 GPT-SoVITS
                async for chunk in self._api_synthesize(text):
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
    
    async def _api_synthesize(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """
        通过 HTTP API 调用 GPT-SoVITS 进行流式合成
        
        API 文档: GPT-SoVITS/api_v2.py
        streaming_mode:
            0: 关闭流式
            1: 最佳质量，最慢响应 (return_fragment=True)
            2: 中等质量，较慢响应 (streaming_mode=True)
            3: 较低质量，快速响应 (streaming_mode=True, fixed_length_chunk=True)
        """
        session = await self._get_session()
        
        # 构建请求参数
        params = {
            "text": text,
            "text_lang": self.config.text_lang.lower(),
            "ref_audio_path": self.config.ref_audio_path,
            "prompt_text": self.config.prompt_text,
            "prompt_lang": self.config.prompt_lang.lower(),
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "speed_factor": self.config.speed_factor,
            "text_split_method": self.config.text_split_method,
            "batch_size": self.config.batch_size,
            "parallel_infer": self.config.parallel_infer,
            "repetition_penalty": self.config.repetition_penalty,
            "streaming_mode": self.config.streaming_mode,
            "overlap_length": self.config.overlap_length,
            "min_chunk_length": self.config.min_chunk_length,
            "fragment_interval": self.config.fragment_interval,
            "media_type": "raw",  # 返回原始 PCM 数据
        }
        
        url = f"{self.config.api_url}/tts"
        logger.debug(f"TTS API request: {url}, text: '{text[:50]}...'")
        
        try:
            start_time = time.time()
            first_chunk_time = None
            total_bytes = 0
            
            async with session.post(url, json=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS API error: {response.status} - {error_text}")
                    # 返回空的最后一块
                    yield AudioChunk(
                        pcm_data=b"",
                        sample_rate=self.config.sample_rate,
                        sequence=0,
                        is_last=True,
                        text=text
                    )
                    return
                
                # 流式读取响应
                chunk_buffer = b""
                chunk_size = 4800  # 约 150ms @ 32kHz, 16bit
                
                async for data in response.content.iter_any():
                    if self._should_stop:
                        logger.info("TTS synthesis interrupted")
                        break
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttfb = (first_chunk_time - start_time) * 1000
                        logger.debug(f"TTS TTFB: {ttfb:.1f}ms")
                    
                    chunk_buffer += data
                    total_bytes += len(data)
                    
                    # 分块输出，保证每个块大小适合播放
                    while len(chunk_buffer) >= chunk_size:
                        pcm_chunk = chunk_buffer[:chunk_size]
                        chunk_buffer = chunk_buffer[chunk_size:]
                        
                        yield AudioChunk(
                            pcm_data=pcm_chunk,
                            sample_rate=self.config.sample_rate,
                            sequence=0,
                            is_last=False,
                            text=text
                        )
                        
                        await asyncio.sleep(0)  # 让出控制权
                
                # 输出剩余数据
                if chunk_buffer and not self._should_stop:
                    yield AudioChunk(
                        pcm_data=chunk_buffer,
                        sample_rate=self.config.sample_rate,
                        sequence=0,
                        is_last=False,
                        text=text
                    )
                
                # 最后一个标记块
                yield AudioChunk(
                    pcm_data=b"",
                    sample_rate=self.config.sample_rate,
                    sequence=0,
                    is_last=True,
                    text=text
                )
                
                total_time = time.time() - start_time
                logger.debug(f"TTS completed: {total_bytes} bytes in {total_time:.2f}s")
                
        except asyncio.TimeoutError:
            logger.error("TTS API request timeout")
            yield AudioChunk(
                pcm_data=b"",
                sample_rate=self.config.sample_rate,
                sequence=0,
                is_last=True,
                text=text
            )
        except aiohttp.ClientError as e:
            logger.error(f"TTS API connection error: {e}")
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
        """模拟合成（用于测试，当 GPT-SoVITS API 不可用时）"""
        logger.debug(f"Mock synthesizing: '{text}'")
        
        # 生成模拟音频（正弦波 + 简单调制）
        duration = len(text) * 0.12  # 每个字符约 120ms
        duration = max(0.3, min(duration, 10.0))  # 限制在 0.3-10 秒
        
        sample_rate = self.config.sample_rate
        total_samples = int(duration * sample_rate)
        chunk_size = int(0.1 * sample_rate)  # 100ms 每块
        
        # 生成正弦波
        t = np.linspace(0, duration, total_samples, dtype=np.float32)
        
        # 使用多个频率模拟语音
        frequency_base = 200  # 基频
        audio = (
            np.sin(2 * np.pi * frequency_base * t) * 0.3 +
            np.sin(2 * np.pi * frequency_base * 2 * t) * 0.15 +
            np.sin(2 * np.pi * frequency_base * 3 * t) * 0.05
        )
        
        # 添加包络（模拟语音的起伏）
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3Hz 调制
        audio = audio * envelope
        
        # 添加淡入淡出
        fade_samples = int(0.02 * sample_rate)  # 20ms
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        # 转换为 int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 分块输出
        offset = 0
        while offset < len(audio_int16):
            if self._should_stop:
                break
            
            end = min(offset + chunk_size, len(audio_int16))
            chunk_data = audio_int16[offset:end]
            is_last = (end >= len(audio_int16))
            
            yield AudioChunk(
                pcm_data=chunk_data.tobytes(),
                sample_rate=sample_rate,
                sequence=0,
                is_last=is_last,
                text=text if is_last else ""
            )
            
            offset = end
            await asyncio.sleep(0.03)  # 模拟合成延迟
    
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
    用于测试（不需要 GPT-SoVITS 服务）
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._is_synthesizing = False
        self._should_stop = False
        self._is_ready = True
        logger.info("MockTTSEngine initialized")
    
    def load_model(self) -> None:
        logger.info("Mock TTS model loaded")
    
    async def close(self) -> None:
        pass
    
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
                
                # 生成模拟音频（类似语音的波形）
                t = np.linspace(0, chunk_duration, chunk_samples, dtype=np.float32)
                freq = 200 + (i % 5) * 50  # 变化频率
                audio = np.sin(2 * np.pi * freq * t) * 0.3
                audio_data = (audio * 32767).astype(np.int16)
                
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


def create_tts_engine(
    use_mock: bool = False, 
    config: Optional[TTSConfig] = None
) -> Union[GPTSoVITSEngine, MockTTSEngine]:
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
        # 测试 Mock 引擎
        print("="*50)
        print("测试 MockTTSEngine")
        print("="*50)
        
        tts = MockTTSEngine()
        tts.load_model()
        
        chunks = []
        async for chunk in tts.synthesize_stream("你好，这是一个测试。"):
            chunks.append(chunk)
            print(f"Chunk #{chunk.sequence}: {len(chunk.pcm_data)} bytes, is_last={chunk.is_last}")
        
        print(f"\nTotal chunks: {len(chunks)}")
        
        # 测试 GPT-SoVITS 引擎（如果服务可用）
        print("\n" + "="*50)
        print("测试 GPTSoVITSEngine")
        print("="*50)
        
        config = TTSConfig(
            api_url="http://127.0.0.1:9880",
            ref_audio_path="/path/to/reference.wav",
            prompt_text="测试参考文本",
            prompt_lang="zh"
        )
        
        tts_real = GPTSoVITSEngine(config)
        tts_real.load_model()
        
        if tts_real._is_ready:
            print("GPT-SoVITS API 服务可用，测试流式合成...")
            chunks = []
            async for chunk in tts_real.synthesize_stream("你好，这是 GPT-SoVITS 测试。"):
                chunks.append(chunk)
                print(f"Chunk #{chunk.sequence}: {len(chunk.pcm_data)} bytes, is_last={chunk.is_last}")
            print(f"\nTotal chunks: {len(chunks)}")
        else:
            print("GPT-SoVITS API 服务不可用，跳过测试")
        
        await tts_real.close()
    
    asyncio.run(test_tts())
