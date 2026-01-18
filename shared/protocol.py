"""
WebSocket 通信协议定义
定义本地客户端与云端服务之间的消息格式
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional
import json
import struct


class MessageType(Enum):
    """消息类型枚举"""
    # 控制消息
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    INTERRUPT = "interrupt"
    INTERRUPT_ACK = "interrupt_ack"
    
    # ASR 相关
    ASR_PARTIAL = "asr_partial"      # 部分识别结果
    ASR_FINAL = "asr_final"          # 最终识别结果
    
    # LLM 相关
    LLM_START = "llm_start"          # LLM 开始生成
    LLM_TOKEN = "llm_token"          # LLM token 流
    LLM_END = "llm_end"              # LLM 生成结束
    
    # TTS 相关
    TTS_CHUNK = "tts_chunk"          # TTS 音频块
    TTS_START = "tts_start"          # TTS 开始
    TTS_END = "tts_end"              # TTS 结束
    
    # 状态消息
    STATUS = "status"
    ERROR = "error"


@dataclass
class TextMessage:
    """文本消息"""
    type: str
    content: str
    sequence: int = 0
    timestamp: float = 0.0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, data: str) -> 'TextMessage':
        return cls(**json.loads(data))


@dataclass
class AudioChunkHeader:
    """音频块头部"""
    sequence: int          # 序列号
    sample_rate: int       # 采样率
    channels: int          # 通道数
    samples: int           # 样本数
    is_last: bool          # 是否最后一块
    
    HEADER_FORMAT = '!IIHIb'  # 网络字节序
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            self.HEADER_FORMAT,
            self.sequence,
            self.sample_rate,
            self.channels,
            self.samples,
            1 if self.is_last else 0
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'AudioChunkHeader':
        seq, sr, ch, samples, is_last = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])
        return cls(seq, sr, ch, samples, bool(is_last))


def create_text_message(msg_type: MessageType, content: str, sequence: int = 0) -> str:
    """创建文本消息的JSON字符串"""
    import time
    return TextMessage(
        type=msg_type.value,
        content=content,
        sequence=sequence,
        timestamp=time.time()
    ).to_json()


def create_audio_frame(header: AudioChunkHeader, pcm_data: bytes) -> bytes:
    """创建音频帧 (头部 + PCM数据)"""
    return header.to_bytes() + pcm_data


def parse_audio_frame(data: bytes) -> tuple[AudioChunkHeader, bytes]:
    """解析音频帧"""
    header = AudioChunkHeader.from_bytes(data)
    pcm_data = data[AudioChunkHeader.HEADER_SIZE:]
    return header, pcm_data


# 常量定义
class AudioConfig:
    """音频配置常量"""
    SAMPLE_RATE = 16000          # 采样率 16kHz
    CHANNELS = 1                 # 单声道
    SAMPLE_WIDTH = 2             # 16-bit
    FRAME_DURATION_MS = 32       # 帧时长 32ms
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 512 samples
    BUFFER_SIZE = 1024           # 缓冲区大小
    PRE_BUFFER_MS = 100          # 预缓冲 100ms
    PRE_BUFFER_FRAMES = PRE_BUFFER_MS // FRAME_DURATION_MS  # 约3帧


class TTSConfig:
    """TTS 配置常量"""
    SAMPLE_RATE = 32000          # 32kHz (GPT-SoVITS v4 输出采样率)
    CHANNELS = 1
    SAMPLE_WIDTH = 2


class NetworkConfig:
    """网络配置常量"""
    WEBSOCKET_PORT = 8765
    HEARTBEAT_INTERVAL = 10.0    # 心跳间隔 10秒
    RECONNECT_BASE_DELAY = 1.0   # 重连基础延迟
    RECONNECT_MAX_DELAY = 60.0   # 重连最大延迟
    RTT_WARNING_THRESHOLD = 0.1  # RTT 警告阈值 100ms
