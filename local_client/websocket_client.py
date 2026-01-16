"""
WebSocket 客户端通信模块
实现与云端服务的实时双向通信
"""

import asyncio
import websockets
from websockets.client import WebSocketClientProtocol
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
import json
import logging
import time
from threading import Thread, Event, Lock
from queue import Queue, Empty
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import (
    MessageType, 
    TextMessage, 
    AudioChunkHeader,
    create_text_message,
    parse_audio_frame,
    NetworkConfig,
    TTSConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStatus:
    """连接状态"""
    is_connected: bool
    server_url: str
    last_rtt_ms: float
    reconnect_count: int
    last_error: Optional[str]


class WebSocketClient:
    """
    WebSocket 客户端
    支持自动重连、心跳检测和低延迟传输
    """
    
    def __init__(
        self,
        server_url: str,
        on_text_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_audio_chunk: Optional[Callable[[AudioChunkHeader, bytes], None]] = None,
        on_connection_change: Optional[Callable[[bool], None]] = None,
        tcp_nodelay: bool = True
    ):
        """
        初始化 WebSocket 客户端
        
        Args:
            server_url: 服务器 URL (ws://host:port)
            on_text_message: 文本消息回调
            on_audio_chunk: 音频块回调
            on_connection_change: 连接状态变化回调
            tcp_nodelay: 是否禁用 Nagle 算法
        """
        self.server_url = server_url
        self._on_text_message = on_text_message
        self._on_audio_chunk = on_audio_chunk
        self._on_connection_change = on_connection_change
        self._tcp_nodelay = tcp_nodelay
        
        # WebSocket 连接
        self._ws: Optional[WebSocketClientProtocol] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None
        
        # 状态
        self._is_running = Event()
        self._is_connected = Event()
        self._reconnect_count = 0
        self._last_rtt_ms = 0.0
        self._last_error: Optional[str] = None
        
        # 发送队列
        self._send_queue: Queue = Queue()
        
        # 心跳
        self._last_heartbeat_time = 0.0
        self._heartbeat_sent_time = 0.0
        
        # 线程安全
        self._lock = Lock()
        
        logger.info(f"WebSocket client initialized: {server_url}")
    
    def start(self) -> None:
        """启动客户端"""
        if self._is_running.is_set():
            logger.warning("WebSocket client already running")
            return
        
        self._is_running.set()
        self._thread = Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        logger.info("WebSocket client started")
    
    def stop(self) -> None:
        """停止客户端"""
        self._is_running.clear()
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        
        logger.info("WebSocket client stopped")
    
    def _run_event_loop(self) -> None:
        """运行事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._loop.close()
    
    async def _connect_loop(self) -> None:
        """连接循环（支持自动重连）"""
        reconnect_delay = NetworkConfig.RECONNECT_BASE_DELAY
        
        while self._is_running.is_set():
            try:
                await self._connect()
                reconnect_delay = NetworkConfig.RECONNECT_BASE_DELAY
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Connection failed: {e}")
                
                if self._is_running.is_set():
                    logger.info(f"Reconnecting in {reconnect_delay:.1f}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2,
                        NetworkConfig.RECONNECT_MAX_DELAY
                    )
                    self._reconnect_count += 1
    
    async def _connect(self) -> None:
        """建立连接"""
        logger.info(f"Connecting to {self.server_url}...")
        
        # 创建自定义 socket 选项
        import socket
        
        async with websockets.connect(
            self.server_url,
            ping_interval=NetworkConfig.HEARTBEAT_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB
            compression=None  # 禁用压缩以降低延迟
        ) as ws:
            # 设置 TCP_NODELAY
            if self._tcp_nodelay:
                sock = ws.transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    logger.debug("TCP_NODELAY enabled")
            
            self._ws = ws
            self._is_connected.set()
            self._last_error = None
            
            logger.info("WebSocket connected")
            
            if self._on_connection_change:
                self._on_connection_change(True)
            
            # 启动任务
            await asyncio.gather(
                self._receive_loop(),
                self._send_loop(),
                self._heartbeat_loop()
            )
    
    async def _receive_loop(self) -> None:
        """接收消息循环"""
        try:
            async for message in self._ws:
                await self._handle_message(message)
        except websockets.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
        finally:
            self._is_connected.clear()
            if self._on_connection_change:
                self._on_connection_change(False)
    
    async def _handle_message(self, message) -> None:
        """处理接收的消息"""
        try:
            if isinstance(message, bytes):
                # 二进制消息（音频数据）
                header, pcm_data = parse_audio_frame(message)
                if self._on_audio_chunk:
                    self._on_audio_chunk(header, pcm_data)
            else:
                # 文本消息（JSON）
                data = json.loads(message)
                msg_type = data.get('type', '')
                
                # 处理心跳响应
                if msg_type == MessageType.HEARTBEAT_ACK.value:
                    self._last_rtt_ms = (time.time() - self._heartbeat_sent_time) * 1000
                    if self._last_rtt_ms > NetworkConfig.RTT_WARNING_THRESHOLD * 1000:
                        logger.warning(f"High RTT detected: {self._last_rtt_ms:.1f}ms")
                else:
                    if self._on_text_message:
                        self._on_text_message(data)
                        
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def _send_loop(self) -> None:
        """发送消息循环"""
        while self._is_connected.is_set():
            try:
                message = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._send_queue.get(timeout=0.1)
                )
                
                if self._ws and self._is_connected.is_set():
                    await self._ws.send(message)
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")
                break
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._is_connected.is_set():
            await asyncio.sleep(NetworkConfig.HEARTBEAT_INTERVAL)
            
            if self._ws and self._is_connected.is_set():
                try:
                    self._heartbeat_sent_time = time.time()
                    await self._ws.send(create_text_message(
                        MessageType.HEARTBEAT, 
                        ""
                    ))
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    break
    
    def send_text(self, msg_type: MessageType, content: str, sequence: int = 0) -> None:
        """
        发送文本消息
        
        Args:
            msg_type: 消息类型
            content: 消息内容
            sequence: 序列号
        """
        if not self._is_connected.is_set():
            logger.warning("Cannot send: not connected")
            return
        
        message = create_text_message(msg_type, content, sequence)
        self._send_queue.put(message)
    
    def send_interrupt(self) -> None:
        """发送打断信号"""
        self.send_text(MessageType.INTERRUPT, "")
        logger.info("Interrupt signal sent")
    
    def send_asr_result(self, text: str, is_final: bool) -> None:
        """
        发送 ASR 结果
        
        Args:
            text: 识别文本
            is_final: 是否为最终结果
        """
        msg_type = MessageType.ASR_FINAL if is_final else MessageType.ASR_PARTIAL
        self.send_text(msg_type, text)
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._is_connected.is_set()
    
    def get_status(self) -> ConnectionStatus:
        """获取连接状态"""
        return ConnectionStatus(
            is_connected=self._is_connected.is_set(),
            server_url=self.server_url,
            last_rtt_ms=self._last_rtt_ms,
            reconnect_count=self._reconnect_count,
            last_error=self._last_error
        )
    
    def wait_connected(self, timeout: float = 10.0) -> bool:
        """
        等待连接建立
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功连接
        """
        return self._is_connected.wait(timeout)


class AudioReceiver:
    """
    音频接收器
    处理从云端接收的 TTS 音频流
    """
    
    def __init__(
        self,
        on_audio_ready: Callable[[np.ndarray], None],
        sample_rate: int = TTSConfig.SAMPLE_RATE
    ):
        """
        初始化音频接收器
        
        Args:
            on_audio_ready: 音频就绪回调
            sample_rate: 采样率
        """
        self._on_audio_ready = on_audio_ready
        self.sample_rate = sample_rate
        
        # 音频块缓冲（按序列号排序）
        self._chunks: Dict[int, bytes] = {}
        self._next_sequence = 0
        self._lock = Lock()
        
        # 状态
        self._is_receiving = False
    
    def on_audio_chunk(self, header: AudioChunkHeader, pcm_data: bytes) -> None:
        """
        处理音频块
        
        Args:
            header: 音频块头部
            pcm_data: PCM 数据
        """
        with self._lock:
            self._is_receiving = True
            
            # 存储块
            self._chunks[header.sequence] = pcm_data
            
            # 按序输出
            while self._next_sequence in self._chunks:
                chunk_data = self._chunks.pop(self._next_sequence)
                
                # 转换为 numpy 数组
                audio = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                self._on_audio_ready(audio)
                self._next_sequence += 1
            
            # 检查是否为最后一块
            if header.is_last:
                self._is_receiving = False
    
    def reset(self) -> None:
        """重置接收器"""
        with self._lock:
            self._chunks.clear()
            self._next_sequence = 0
            self._is_receiving = False
    
    @property
    def is_receiving(self) -> bool:
        """是否正在接收"""
        return self._is_receiving


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    def on_text(data):
        print(f"Text message: {data}")
    
    def on_audio(header, data):
        print(f"Audio chunk: seq={header.sequence}, samples={header.samples}")
    
    def on_connection(connected):
        print(f"Connection status: {'connected' if connected else 'disconnected'}")
    
    # 创建客户端
    client = WebSocketClient(
        server_url="ws://localhost:8765",
        on_text_message=on_text,
        on_audio_chunk=on_audio,
        on_connection_change=on_connection
    )
    
    print("Starting WebSocket client...")
    client.start()
    
    # 等待连接
    if client.wait_connected(5.0):
        print("Connected!")
        
        # 发送测试消息
        client.send_text(MessageType.ASR_FINAL, "你好，这是一个测试")
        
        time.sleep(2)
    else:
        print("Connection timeout")
    
    client.stop()
    print("Test complete")
