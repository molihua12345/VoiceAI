"""
系统配置管理
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioCaptureConfig:
    """音频采集配置"""
    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 32
    pre_buffer_ms: int = 100
    device_index: Optional[int] = None


@dataclass
class VADConfig:
    """VAD 配置"""
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 30
    use_gpu: bool = True


@dataclass
class ASRConfig:
    """ASR 配置"""
    model_path: Optional[str] = None  # None 表示自动下载
    language: str = "zh"
    use_itn: bool = True
    use_gpu: bool = True
    min_audio_length_ms: int = 500
    max_audio_length_ms: int = 10000


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = "Qwen/Qwen2-7B-Instruct-AWQ"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    gpu_memory_utilization: float = 0.85
    system_prompt: str = "你是一个友好的AI助手，请用简洁自然的语言回答问题。"


@dataclass
class TTSConfig:
    """TTS 配置"""
    model_path: str = ""
    ref_audio_path: str = ""
    ref_text: str = ""
    sample_rate: int = 32000
    streaming_mode: int = 3
    overlap_length: int = 2
    speed: float = 1.0
    language: str = "zh"


@dataclass
class NetworkConfig:
    """网络配置"""
    server_host: str = "0.0.0.0"
    server_port: int = 8765
    client_server_url: str = "ws://localhost:8765"
    heartbeat_interval: float = 10.0
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    tcp_nodelay: bool = True


@dataclass
class LocalClientConfig:
    """本地客户端配置"""
    audio: AudioCaptureConfig = field(default_factory=AudioCaptureConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class CloudServerConfig:
    """云端服务配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class SystemConfig:
    """系统总配置"""
    local_client: LocalClientConfig = field(default_factory=LocalClientConfig)
    cloud_server: CloudServerConfig = field(default_factory=CloudServerConfig)
    use_mock: bool = False
    debug: bool = False


def load_config(config_path: str) -> SystemConfig:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        SystemConfig: 系统配置
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return SystemConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = SystemConfig()
        
        # 解析本地客户端配置
        if 'local_client' in data:
            lc = data['local_client']
            if 'audio' in lc:
                config.local_client.audio = AudioCaptureConfig(**lc['audio'])
            if 'vad' in lc:
                config.local_client.vad = VADConfig(**lc['vad'])
            if 'asr' in lc:
                config.local_client.asr = ASRConfig(**lc['asr'])
            if 'network' in lc:
                config.local_client.network = NetworkConfig(**lc['network'])
        
        # 解析云端服务配置
        if 'cloud_server' in data:
            cs = data['cloud_server']
            if 'llm' in cs:
                config.cloud_server.llm = LLMConfig(**cs['llm'])
            if 'tts' in cs:
                config.cloud_server.tts = TTSConfig(**cs['tts'])
            if 'network' in cs:
                config.cloud_server.network = NetworkConfig(**cs['network'])
        
        # 全局配置
        config.use_mock = data.get('use_mock', False)
        config.debug = data.get('debug', False)
        
        logger.info(f"Config loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return SystemConfig()


def save_config(config: SystemConfig, config_path: str) -> None:
    """
    保存配置到 YAML 文件
    
    Args:
        config: 系统配置
        config_path: 配置文件路径
    """
    try:
        data = {
            'use_mock': config.use_mock,
            'debug': config.debug,
            'local_client': {
                'audio': asdict(config.local_client.audio),
                'vad': asdict(config.local_client.vad),
                'asr': asdict(config.local_client.asr),
                'network': asdict(config.local_client.network),
            },
            'cloud_server': {
                'llm': asdict(config.cloud_server.llm),
                'tts': asdict(config.cloud_server.tts),
                'network': asdict(config.cloud_server.network),
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Config saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


def get_default_config() -> SystemConfig:
    """获取默认配置"""
    return SystemConfig()


if __name__ == "__main__":
    # 生成默认配置文件
    config = get_default_config()
    save_config(config, "config.yaml")
    print("Default config saved to config.yaml")
