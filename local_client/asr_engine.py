"""
ASR 引擎模块
基于 SenseVoiceSmall 实现本地流式语音识别
"""

import numpy as np
import torch
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from threading import Lock, Thread
from queue import Queue, Empty
import logging
import time
import os

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str                 # 识别文本
    is_partial: bool          # 是否为部分结果
    confidence: float         # 置信度
    latency_ms: float         # 延迟（毫秒）
    timestamp: float          # 时间戳


class SenseVoiceASR:
    """
    基于 SenseVoiceSmall 的 ASR 引擎
    支持流式和批量识别
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        language: str = "zh",
        use_itn: bool = True
    ):
        """
        初始化 ASR 引擎
        
        Args:
            model_path: 模型路径（如果为None，则从 ModelScope 下载）
            use_gpu: 是否使用 GPU
            language: 语言 (zh, en, ja, ko, yue)
            use_itn: 是否使用逆文本标准化
        """
        self.language = language
        self.use_itn = use_itn
        
        # 设备选择
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info("ASR using GPU acceleration")
        else:
            self.device = "cpu"
            logger.info("ASR using CPU")
        
        # 加载模型
        self._load_model(model_path)
        
        # 线程安全
        self._lock = Lock()
        
        # 统计
        self._total_latency = 0.0
        self._inference_count = 0
        
        logger.info(f"SenseVoiceASR initialized: language={language}")
    
    def _load_model(self, model_path: Optional[str]) -> None:
        """加载 SenseVoice 模型"""
        # 设置纯英文缓存路径，避免中文路径问题
        self._setup_cache_path()
        
        try:
            from funasr import AutoModel
            
            # 如果没有指定路径，使用 ModelScope 模型
            if model_path is None:
                model_path = "iic/SenseVoiceSmall"
            
            logger.info(f"Loading SenseVoice model from: {model_path}")
            
            self.model = AutoModel(
                model=model_path,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
            )
            
            logger.info("SenseVoice model loaded successfully")
            
        except ImportError as e:
            logger.error(f"FunASR not installed. Please install: pip install funasr")
            raise RuntimeError(f"ASR dependency missing: {e}")
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model: {e}")
            # 检查 GPU 内存
            if torch.cuda.is_available():
                try:
                    memory_info = torch.cuda.memory_summary()
                    logger.error(f"GPU Memory Status:\n{memory_info}")
                except:
                    pass
            raise RuntimeError(f"ASR model loading failed: {e}")
    
    def _setup_cache_path(self) -> None:
        """设置缓存路径到纯英文目录，避免中文路径问题"""
        user_home = os.path.expanduser('~')
        
        def has_non_ascii(path: str) -> bool:
            try:
                path.encode('ascii')
                return False
            except UnicodeEncodeError:
                return True
        
        # 如果用户目录包含中文，设置缓存到纯英文路径
        if has_non_ascii(user_home):
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 优先使用项目目录
            if not has_non_ascii(project_dir):
                cache_base = os.path.join(project_dir, '.model_cache')
            else:
                cache_base = 'C:/model_cache'
            
            # 设置 ModelScope 缓存
            modelscope_cache = os.path.join(cache_base, 'modelscope')
            os.makedirs(modelscope_cache, exist_ok=True)
            os.environ['MODELSCOPE_CACHE'] = modelscope_cache
            
            # 设置 HuggingFace 缓存
            hf_cache = os.path.join(cache_base, 'huggingface')
            os.makedirs(hf_cache, exist_ok=True)
            os.environ['HF_HOME'] = hf_cache
            
            logger.info(f"Set model cache to ASCII path: {cache_base}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = AudioConfig.SAMPLE_RATE
    ) -> ASRResult:
        """
        转写音频
        
        Args:
            audio: float32 格式的音频数据
            sample_rate: 采样率
        
        Returns:
            ASRResult: 识别结果
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # 确保音频是正确的格式
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                # 调用模型
                result = self.model.generate(
                    input=audio,
                    cache={},
                    language=self.language,
                    use_itn=self.use_itn,
                    batch_size_s=60,
                )
                
                # 解析结果
                if result and len(result) > 0:
                    text = result[0].get("text", "")
                    # 清理文本（移除特殊标记）
                    text = self._clean_text(text)
                else:
                    text = ""
                
                latency_ms = (time.time() - start_time) * 1000
                self._total_latency += latency_ms
                self._inference_count += 1
                
                logger.debug(f"ASR result: '{text}' (latency: {latency_ms:.1f}ms)")
                
                return ASRResult(
                    text=text,
                    is_partial=False,
                    confidence=1.0,
                    latency_ms=latency_ms,
                    timestamp=time.time()
                )
                
            except Exception as e:
                logger.error(f"ASR transcription error: {e}")
                return ASRResult(
                    text="",
                    is_partial=False,
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time()
                )
    
    def _clean_text(self, text: str) -> str:
        """清理识别文本，移除特殊标记"""
        import re
        # 移除 SenseVoice 的特殊标记 <|xx|>
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()
    
    @property
    def average_latency(self) -> float:
        """平均推理延迟（毫秒）"""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency / self._inference_count


class StreamingASR:
    """
    流式 ASR 处理器
    支持增量识别和异步处理
    """
    
    def __init__(
        self,
        asr_engine: SenseVoiceASR,
        on_partial_result: Optional[Callable[[str], None]] = None,
        on_final_result: Optional[Callable[[str], None]] = None,
        min_audio_length_ms: int = 500,
        max_audio_length_ms: int = 10000
    ):
        """
        初始化流式 ASR
        
        Args:
            asr_engine: ASR 引擎
            on_partial_result: 部分结果回调
            on_final_result: 最终结果回调
            min_audio_length_ms: 最小音频长度触发识别
            max_audio_length_ms: 最大音频长度
        """
        self.asr = asr_engine
        self._on_partial_result = on_partial_result
        self._on_final_result = on_final_result
        
        self.min_audio_length = int(AudioConfig.SAMPLE_RATE * min_audio_length_ms / 1000)
        self.max_audio_length = int(AudioConfig.SAMPLE_RATE * max_audio_length_ms / 1000)
        
        # 音频缓冲
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_lock = Lock()
        
        # 异步处理
        self._processing_queue: Queue = Queue()
        self._is_running = False
        self._process_thread: Optional[Thread] = None
        
        # 上一次的识别结果（用于增量输出）
        self._last_text = ""
    
    def start(self) -> None:
        """启动流式处理"""
        if self._is_running:
            return
        
        self._is_running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        logger.info("Streaming ASR started")
    
    def stop(self) -> None:
        """停止流式处理"""
        self._is_running = False
        if self._process_thread:
            self._processing_queue.put(None)  # 发送停止信号
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
        logger.info("Streaming ASR stopped")
    
    def add_audio(self, audio_frame: np.ndarray) -> None:
        """
        添加音频帧
        
        Args:
            audio_frame: 音频帧
        """
        with self._buffer_lock:
            self._audio_buffer.append(audio_frame.copy())
    
    def process_accumulated(self, is_final: bool = False) -> Optional[str]:
        """
        处理累积的音频
        
        Args:
            is_final: 是否为最终处理
        
        Returns:
            识别文本或 None
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            
            audio = np.concatenate(self._audio_buffer)
            
            # 检查音频长度
            if not is_final and len(audio) < self.min_audio_length:
                return None
            
            # 截断过长音频
            if len(audio) > self.max_audio_length:
                audio = audio[-self.max_audio_length:]
            
            if is_final:
                self._audio_buffer = []
        
        # 执行识别
        result = self.asr.transcribe(audio)
        
        if result.text:
            if is_final:
                self._last_text = ""
                if self._on_final_result:
                    self._on_final_result(result.text)
                return result.text
            else:
                # 增量输出
                if result.text != self._last_text:
                    new_text = result.text
                    self._last_text = result.text
                    if self._on_partial_result:
                        self._on_partial_result(new_text)
                    return new_text
        
        return None
    
    def finalize(self) -> Optional[str]:
        """完成当前语音段的识别"""
        return self.process_accumulated(is_final=True)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._buffer_lock:
            self._audio_buffer = []
        self._last_text = ""
    
    def _process_loop(self) -> None:
        """异步处理循环"""
        while self._is_running:
            try:
                task = self._processing_queue.get(timeout=0.1)
                if task is None:
                    break
                # 处理任务
                audio, is_final = task
                self._do_process(audio, is_final)
            except Empty:
                continue
    
    def _do_process(self, audio: np.ndarray, is_final: bool) -> None:
        """执行处理"""
        result = self.asr.transcribe(audio)
        
        if result.text:
            if is_final and self._on_final_result:
                self._on_final_result(result.text)
            elif not is_final and self._on_partial_result:
                self._on_partial_result(result.text)
    
    def get_buffer_duration_ms(self) -> float:
        """获取缓冲区音频时长（毫秒）"""
        with self._buffer_lock:
            if not self._audio_buffer:
                return 0.0
            total_samples = sum(len(f) for f in self._audio_buffer)
            return total_samples / AudioConfig.SAMPLE_RATE * 1000


class ASRManager:
    """
    ASR 管理器
    整合 VAD 和 ASR，提供完整的语音识别流程
    """
    
    def __init__(
        self,
        on_transcription: Optional[Callable[[str, bool], None]] = None,
        **asr_kwargs
    ):
        """
        初始化 ASR 管理器
        
        Args:
            on_transcription: 转写结果回调 (text, is_final)
            **asr_kwargs: ASR 引擎参数
        """
        self._on_transcription = on_transcription
        
        # 创建 ASR 引擎
        self.asr_engine = SenseVoiceASR(**asr_kwargs)
        
        # 创建流式处理器
        self.streaming = StreamingASR(
            asr_engine=self.asr_engine,
            on_partial_result=lambda t: self._handle_result(t, False),
            on_final_result=lambda t: self._handle_result(t, True)
        )
        
        # 状态
        self._is_active = False
    
    def _handle_result(self, text: str, is_final: bool) -> None:
        """处理识别结果"""
        if self._on_transcription:
            self._on_transcription(text, is_final)
    
    def start_recognition(self, pre_buffer_audio: Optional[np.ndarray] = None) -> None:
        """
        开始识别
        
        Args:
            pre_buffer_audio: 预缓冲的音频（语音起始部分）
        """
        self._is_active = True
        self.streaming.clear()
        
        if pre_buffer_audio is not None and len(pre_buffer_audio) > 0:
            self.streaming.add_audio(pre_buffer_audio)
        
        logger.debug("ASR recognition started")
    
    def add_audio(self, audio_frame: np.ndarray) -> None:
        """添加音频帧"""
        if self._is_active:
            self.streaming.add_audio(audio_frame)
    
    def end_recognition(self) -> Optional[str]:
        """
        结束识别并返回最终结果
        
        Returns:
            最终识别文本
        """
        self._is_active = False
        result = self.streaming.finalize()
        logger.debug(f"ASR recognition ended: '{result}'")
        return result
    
    def cancel_recognition(self) -> None:
        """取消当前识别"""
        self._is_active = False
        self.streaming.clear()
        logger.debug("ASR recognition cancelled")
    
    @property
    def is_active(self) -> bool:
        """是否正在识别"""
        return self._is_active


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ASR Engine...")
    
    try:
        # 创建 ASR 引擎
        asr = SenseVoiceASR()
        
        # 生成测试音频（静音）
        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        
        # 测试转写
        result = asr.transcribe(test_audio)
        print(f"Result: '{result.text}' (latency: {result.latency_ms:.1f}ms)")
        
    except Exception as e:
        print(f"ASR test failed: {e}")
        print("\nTo use ASR, please install FunASR:")
        print("  pip install funasr modelscope")
