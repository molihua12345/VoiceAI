"""
WebSocket 服务端
处理客户端连接和消息路由
"""

import asyncio
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Dict, Set, Optional, Callable, Any
import json
import logging
import time
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import (
    MessageType,
    TextMessage,
    AudioChunkHeader,
    create_text_message,
    create_audio_frame,
    NetworkConfig,
    TTSConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ClientSession:
    """客户端会话"""
    websocket: WebSocketServerProtocol
    client_id: str
    connected_at: float
    last_message_at: float
    is_processing: bool = False


class WebSocketServer:
    """
    WebSocket 服务端
    管理客户端连接和消息处理
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = NetworkConfig.WEBSOCKET_PORT,
        on_text_received: Optional[Callable[[str, str, dict], asyncio.coroutine]] = None,
        on_client_connect: Optional[Callable[[str], None]] = None,
        on_client_disconnect: Optional[Callable[[str], None]] = None
    ):
        """
        初始化 WebSocket 服务端
        
        Args:
            host: 绑定地址
            port: 端口
            on_text_received: 文本消息回调 (client_id, msg_type, data)
            on_client_connect: 客户端连接回调
            on_client_disconnect: 客户端断开回调
        """
        self.host = host
        self.port = port
        self._on_text_received = on_text_received
        self._on_client_connect = on_client_connect
        self._on_client_disconnect = on_client_disconnect
        
        # 客户端会话
        self._sessions: Dict[str, ClientSession] = {}
        self._client_counter = 0
        
        # 服务器实例
        self._server = None
        self._is_running = False
        
        logger.info(f"WebSocket server initialized: {host}:{port}")
    
    async def start(self) -> None:
        """启动服务器"""
        if self._is_running:
            logger.warning("Server already running")
            return
        
        self._is_running = True
        
        # 启动 WebSocket 服务器
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=NetworkConfig.HEARTBEAT_INTERVAL,
            ping_timeout=30,
            close_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB
            compression=None  # 禁用压缩以降低延迟
        )
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """停止服务器"""
        self._is_running = False
        
        # 关闭所有连接
        for session in list(self._sessions.values()):
            await session.websocket.close()
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """处理客户端连接"""
        # 生成客户端 ID
        self._client_counter += 1
        client_id = f"client_{self._client_counter}"
        
        # 创建会话
        session = ClientSession(
            websocket=websocket,
            client_id=client_id,
            connected_at=time.time(),
            last_message_at=time.time()
        )
        self._sessions[client_id] = session
        
        remote_addr = websocket.remote_address
        logger.info(f"Client connected: {client_id} from {remote_addr}")
        
        if self._on_client_connect:
            self._on_client_connect(client_id)
        
        try:
            async for message in websocket:
                session.last_message_at = time.time()
                await self._handle_message(client_id, message)
                
        except websockets.ConnectionClosed as e:
            logger.info(f"Client disconnected: {client_id} ({e.code})")
        except Exception as e:
            logger.error(f"Connection error for {client_id}: {e}")
        finally:
            # 清理会话
            if client_id in self._sessions:
                del self._sessions[client_id]
            
            if self._on_client_disconnect:
                self._on_client_disconnect(client_id)
    
    async def _handle_message(self, client_id: str, message) -> None:
        """处理消息"""
        try:
            if isinstance(message, bytes):
                # 二进制消息（暂时不处理）
                logger.debug(f"Binary message from {client_id}: {len(message)} bytes")
            else:
                # JSON 文本消息
                data = json.loads(message)
                msg_type = data.get('type', '')
                
                # 处理心跳
                if msg_type == MessageType.HEARTBEAT.value:
                    await self.send_text(
                        client_id,
                        MessageType.HEARTBEAT_ACK,
                        ""
                    )
                    return
                
                # 调用回调处理其他消息
                if self._on_text_received:
                    await self._on_text_received(client_id, msg_type, data)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {client_id}: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def send_text(
        self,
        client_id: str,
        msg_type: MessageType,
        content: str,
        sequence: int = 0
    ) -> bool:
        """
        发送文本消息
        
        Args:
            client_id: 客户端 ID
            msg_type: 消息类型
            content: 消息内容
            sequence: 序列号
        
        Returns:
            是否发送成功
        """
        session = self._sessions.get(client_id)
        if not session:
            logger.warning(f"Client not found: {client_id}")
            return False
        
        try:
            message = create_text_message(msg_type, content, sequence)
            await session.websocket.send(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
            return False
    
    async def send_audio(
        self,
        client_id: str,
        pcm_data: bytes,
        sequence: int,
        sample_rate: int = TTSConfig.SAMPLE_RATE,
        is_last: bool = False
    ) -> bool:
        """
        发送音频块
        
        Args:
            client_id: 客户端 ID
            pcm_data: PCM 数据
            sequence: 序列号
            sample_rate: 采样率
            is_last: 是否最后一块
        
        Returns:
            是否发送成功
        """
        session = self._sessions.get(client_id)
        if not session:
            return False
        
        try:
            # 创建音频帧
            header = AudioChunkHeader(
                sequence=sequence,
                sample_rate=sample_rate,
                channels=1,
                samples=len(pcm_data) // 2,  # int16
                is_last=is_last
            )
            
            frame = create_audio_frame(header, pcm_data)
            await session.websocket.send(frame)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send audio to {client_id}: {e}")
            return False
    
    async def broadcast_text(
        self,
        msg_type: MessageType,
        content: str
    ) -> None:
        """广播文本消息到所有客户端"""
        message = create_text_message(msg_type, content)
        
        for session in self._sessions.values():
            try:
                await session.websocket.send(message)
            except Exception as e:
                logger.error(f"Broadcast error to {session.client_id}: {e}")
    
    def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """获取客户端会话"""
        return self._sessions.get(client_id)
    
    def set_client_processing(self, client_id: str, is_processing: bool) -> None:
        """设置客户端处理状态"""
        session = self._sessions.get(client_id)
        if session:
            session.is_processing = is_processing
    
    @property
    def client_count(self) -> int:
        """当前连接的客户端数"""
        return len(self._sessions)
    
    @property
    def client_ids(self) -> list:
        """所有客户端 ID"""
        return list(self._sessions.keys())


class MessageRouter:
    """
    消息路由器
    处理不同类型的消息并路由到相应的处理器
    """
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
    
    def register(self, msg_type: str, handler: Callable) -> None:
        """注册消息处理器"""
        self._handlers[msg_type] = handler
        logger.debug(f"Registered handler for: {msg_type}")
    
    async def route(self, client_id: str, msg_type: str, data: dict) -> Any:
        """路由消息到处理器"""
        handler = self._handlers.get(msg_type)
        
        if handler:
            return await handler(client_id, data)
        else:
            logger.warning(f"No handler for message type: {msg_type}")
            return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    async def on_text(client_id: str, msg_type: str, data: dict):
        print(f"Message from {client_id}: {msg_type} - {data}")
        
    async def main():
        server = WebSocketServer(
            host="localhost",
            port=8765,
            on_text_received=on_text
        )
        
        await server.start()
        
        print(f"Server running on ws://localhost:8765")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        
        await server.stop()
    
    asyncio.run(main())
