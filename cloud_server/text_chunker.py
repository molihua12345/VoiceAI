"""
文本微块化处理模块
实现 LLM 输出的智能分块，用于触发 TTS 合成
"""

import re
from typing import Generator, Optional, Callable, List
from dataclasses import dataclass
from threading import Lock
import logging
import time

logger = logging.getLogger(__name__)


# 标点符号集合（用于分块）
PUNCTUATION_MARKS = set(',.\!?，。！？\n、；：;:')
SENTENCE_END_MARKS = set('.\!?。！？\n')


@dataclass
class TextChunk:
    """文本块"""
    text: str
    sequence: int
    is_sentence_end: bool
    timestamp: float


class TextChunker:
    """
    文本分块器
    将 LLM 流式输出按标点符号切分为微块
    """
    
    def __init__(
        self,
        max_chunk_length: int = 20,
        min_chunk_length: int = 2,
        on_chunk_ready: Optional[Callable[[TextChunk], None]] = None
    ):
        """
        初始化分块器
        
        Args:
            max_chunk_length: 最大块长度（无标点时强制分割）
            min_chunk_length: 最小块长度
            on_chunk_ready: 块就绪回调
        """
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self._on_chunk_ready = on_chunk_ready
        
        # 缓冲区
        self._buffer = ""
        self._sequence = 0
        self._lock = Lock()
        
        logger.info(f"TextChunker initialized: max_len={max_chunk_length}")
    
    def add_token(self, token: str) -> Optional[TextChunk]:
        """
        添加 token
        
        Args:
            token: LLM 生成的 token
        
        Returns:
            如果产生了块，返回 TextChunk，否则返回 None
        """
        with self._lock:
            self._buffer += token
            
            # 检查是否应该输出块
            chunk = self._try_emit_chunk()
            
            if chunk and self._on_chunk_ready:
                self._on_chunk_ready(chunk)
            
            return chunk
    
    def _try_emit_chunk(self) -> Optional[TextChunk]:
        """尝试输出块"""
        if not self._buffer:
            return None
        
        # 查找标点符号位置
        punctuation_pos = -1
        is_sentence_end = False
        
        for i, char in enumerate(self._buffer):
            if char in PUNCTUATION_MARKS:
                punctuation_pos = i
                is_sentence_end = char in SENTENCE_END_MARKS
                break
        
        # 有标点符号，在标点处分割
        if punctuation_pos >= 0:
            chunk_text = self._buffer[:punctuation_pos + 1]
            self._buffer = self._buffer[punctuation_pos + 1:]
            
            # 检查最小长度
            if len(chunk_text.strip()) >= self.min_chunk_length:
                return self._create_chunk(chunk_text, is_sentence_end)
            else:
                # 太短，继续累积
                return None
        
        # 没有标点，检查是否超过最大长度
        if len(self._buffer) >= self.max_chunk_length:
            # 强制分割
            chunk_text = self._buffer
            self._buffer = ""
            return self._create_chunk(chunk_text, False)
        
        return None
    
    def _create_chunk(self, text: str, is_sentence_end: bool) -> TextChunk:
        """创建块"""
        chunk = TextChunk(
            text=text,
            sequence=self._sequence,
            is_sentence_end=is_sentence_end,
            timestamp=time.time()
        )
        self._sequence += 1
        logger.debug(f"Chunk #{chunk.sequence}: '{text}' (sentence_end={is_sentence_end})")
        return chunk
    
    def flush(self) -> Optional[TextChunk]:
        """
        刷新缓冲区，输出剩余文本
        
        Returns:
            最后一个块，如果有的话
        """
        with self._lock:
            if self._buffer and len(self._buffer.strip()) > 0:
                chunk_text = self._buffer
                self._buffer = ""
                chunk = self._create_chunk(chunk_text, True)
                
                if self._on_chunk_ready:
                    self._on_chunk_ready(chunk)
                
                return chunk
            
            return None
    
    def reset(self) -> None:
        """重置分块器"""
        with self._lock:
            self._buffer = ""
            self._sequence = 0
    
    @property
    def buffer_length(self) -> int:
        """当前缓冲区长度"""
        with self._lock:
            return len(self._buffer)


class StreamingTextProcessor:
    """
    流式文本处理器
    处理 LLM 输出并生成 TTS 输入块
    """
    
    def __init__(
        self,
        on_chunk: Optional[Callable[[TextChunk], None]] = None,
        max_chunk_length: int = 20,
        filter_special_tokens: bool = True
    ):
        """
        初始化处理器
        
        Args:
            on_chunk: 块回调
            max_chunk_length: 最大块长度
            filter_special_tokens: 是否过滤特殊 token
        """
        self._on_chunk = on_chunk
        self._filter_special = filter_special_tokens
        
        # 分块器
        self.chunker = TextChunker(
            max_chunk_length=max_chunk_length,
            on_chunk_ready=self._handle_chunk
        )
        
        # 累积的完整文本
        self._full_text = ""
        self._lock = Lock()
    
    def process_token(self, token: str) -> None:
        """
        处理 token
        
        Args:
            token: LLM token
        """
        # 过滤特殊 token
        if self._filter_special:
            token = self._filter_special_tokens(token)
        
        if not token:
            return
        
        with self._lock:
            self._full_text += token
        
        self.chunker.add_token(token)
    
    def _filter_special_tokens(self, token: str) -> str:
        """过滤特殊 token"""
        # 移除常见的特殊标记
        special_patterns = [
            r'<\|[^|]+\|>',  # <|xxx|>
            r'\[.*?\]',      # [xxx]
        ]
        
        for pattern in special_patterns:
            token = re.sub(pattern, '', token)
        
        return token
    
    def _handle_chunk(self, chunk: TextChunk) -> None:
        """处理块"""
        if self._on_chunk:
            self._on_chunk(chunk)
    
    def finalize(self) -> Optional[TextChunk]:
        """完成处理，刷新剩余内容"""
        return self.chunker.flush()
    
    def reset(self) -> None:
        """重置处理器"""
        self.chunker.reset()
        with self._lock:
            self._full_text = ""
    
    def get_full_text(self) -> str:
        """获取完整文本"""
        with self._lock:
            return self._full_text


async def process_llm_stream(
    token_stream,
    on_chunk: Callable[[TextChunk], None],
    max_chunk_length: int = 20
) -> str:
    """
    处理 LLM 流式输出
    
    Args:
        token_stream: LLM token 异步生成器
        on_chunk: 块回调
        max_chunk_length: 最大块长度
    
    Returns:
        完整的生成文本
    """
    processor = StreamingTextProcessor(
        on_chunk=on_chunk,
        max_chunk_length=max_chunk_length
    )
    
    try:
        async for token in token_stream:
            processor.process_token(token)
    finally:
        processor.finalize()
    
    return processor.get_full_text()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    chunks = []
    
    def on_chunk(chunk):
        chunks.append(chunk)
        print(f"Chunk: '{chunk.text}'")
    
    chunker = TextChunker(on_chunk_ready=on_chunk)
    
    # 模拟 LLM 输出
    text = "你好，我是AI助手。很高兴认识你！今天天气真不错，我们聊聊吧。"
    
    print("Input:", text)
    print("\nProcessing token by token:")
    
    for char in text:
        chunker.add_token(char)
    
    # 刷新剩余
    chunker.flush()
    
    print(f"\nTotal chunks: {len(chunks)}")
