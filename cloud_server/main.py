"""
云端服务主程序
整合 LLM、TTS 和 WebSocket 服务
"""

import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging
import time
import argparse

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from shared.protocol import MessageType, NetworkConfig, TTSConfig as TTSProtocolConfig
from cloud_server.llm_engine import (
    LLMEngine, 
    MockLLMEngine, 
    LLMConfig, 
    ConversationContext,
    create_llm_engine
)
from cloud_server.text_chunker import TextChunker, TextChunk, StreamingTextProcessor
from cloud_server.tts_engine import (
    GPTSoVITSEngine,
    MockTTSEngine,
    TTSConfig,
    AudioChunk,
    create_tts_engine
)
from cloud_server.websocket_server import WebSocketServer, MessageRouter

logger = logging.getLogger(__name__)


@dataclass
class ClientContext:
    """客户端上下文"""
    client_id: str
    conversation: ConversationContext
    is_generating: bool = False
    is_interrupted: bool = False
    current_task: Optional[asyncio.Task] = None


class VoiceAIServer:
    """
    实时语音对话云端服务
    处理 LLM 推理和 TTS 合成
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = NetworkConfig.WEBSOCKET_PORT,
        use_mock: bool = False,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None
    ):
        """
        初始化服务
        
        Args:
            host: 绑定地址
            port: 端口
            use_mock: 是否使用模拟模式
            llm_config: LLM 配置
            tts_config: TTS 配置
        """
        self.host = host
        self.port = port
        self.use_mock = use_mock
        
        # 客户端上下文
        self._clients: Dict[str, ClientContext] = {}
        
        # 创建引擎
        self.llm = create_llm_engine(use_mock, llm_config)
        self.tts = create_tts_engine(use_mock, tts_config)
        
        # 创建 WebSocket 服务器
        self.ws_server = WebSocketServer(
            host=host,
            port=port,
            on_text_received=self._on_message,
            on_client_connect=self._on_client_connect,
            on_client_disconnect=self._on_client_disconnect
        )
        
        # 消息路由
        self.router = MessageRouter()
        self._setup_routes()
        
        # 统计
        self._total_requests = 0
        self._start_time = 0.0
        
        logger.info(f"VoiceAI Server initialized (mock={use_mock})")
    
    def _setup_routes(self) -> None:
        """设置消息路由"""
        self.router.register(MessageType.ASR_PARTIAL.value, self._handle_asr_partial)
        self.router.register(MessageType.ASR_FINAL.value, self._handle_asr_final)
        self.router.register(MessageType.INTERRUPT.value, self._handle_interrupt)
    
    async def start(self) -> None:
        """启动服务"""
        logger.info("Starting VoiceAI Server...")
        self._start_time = time.time()
        
        # 加载模型
        logger.info("Loading models...")
        self.llm.load_model()
        self.tts.load_model()
        
        # 启动 WebSocket 服务器
        await self.ws_server.start()
        
        logger.info(f"VoiceAI Server running on ws://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """停止服务"""
        logger.info("Stopping VoiceAI Server...")
        
        # 取消所有进行中的任务
        for ctx in self._clients.values():
            if ctx.current_task and not ctx.current_task.done():
                ctx.current_task.cancel()
        
        await self.ws_server.stop()
        
        uptime = time.time() - self._start_time
        logger.info(f"Server stopped. Uptime: {uptime:.1f}s, Requests: {self._total_requests}")
    
    def _on_client_connect(self, client_id: str) -> None:
        """客户端连接回调"""
        self._clients[client_id] = ClientContext(
            client_id=client_id,
            conversation=ConversationContext()
        )
        logger.info(f"Client context created: {client_id}")
    
    def _on_client_disconnect(self, client_id: str) -> None:
        """客户端断开回调"""
        ctx = self._clients.pop(client_id, None)
        if ctx and ctx.current_task and not ctx.current_task.done():
            ctx.current_task.cancel()
        logger.info(f"Client context removed: {client_id}")
    
    async def _on_message(self, client_id: str, msg_type: str, data: dict) -> None:
        """消息处理回调"""
        await self.router.route(client_id, msg_type, data)
    
    async def _handle_asr_partial(self, client_id: str, data: dict) -> None:
        """处理部分 ASR 结果"""
        text = data.get('content', '')
        logger.debug(f"ASR partial from {client_id}: '{text}'")
        # 部分结果暂不处理，等待最终结果
    
    async def _handle_asr_final(self, client_id: str, data: dict) -> None:
        """处理最终 ASR 结果"""
        text = data.get('content', '').strip()
        
        if not text:
            logger.debug(f"Empty ASR result from {client_id}")
            return
        
        logger.info(f"ASR final from {client_id}: '{text}'")
        
        ctx = self._clients.get(client_id)
        if not ctx:
            logger.warning(f"No context for client: {client_id}")
            return
        
        # 如果正在生成，先中断
        if ctx.is_generating:
            await self._do_interrupt(client_id)
        
        # 添加用户消息到上下文
        ctx.conversation.add_user_message(text)
        
        # 开始生成响应
        self._total_requests += 1
        ctx.current_task = asyncio.create_task(
            self._generate_response(client_id, text)
        )
    
    async def _handle_interrupt(self, client_id: str, data: dict) -> None:
        """处理打断信号"""
        logger.info(f"Interrupt received from {client_id}")
        await self._do_interrupt(client_id)
        
        # 发送打断确认
        await self.ws_server.send_text(
            client_id,
            MessageType.INTERRUPT_ACK,
            ""
        )
    
    async def _do_interrupt(self, client_id: str) -> None:
        """执行打断"""
        ctx = self._clients.get(client_id)
        if not ctx:
            return
        
        ctx.is_interrupted = True
        
        # 中断 LLM
        self.llm.interrupt()
        
        # 中断 TTS
        self.tts.interrupt()
        
        # 取消当前任务
        if ctx.current_task and not ctx.current_task.done():
            ctx.current_task.cancel()
            try:
                await ctx.current_task
            except asyncio.CancelledError:
                pass
        
        ctx.is_generating = False
        ctx.is_interrupted = False
        
        logger.info(f"Interrupt completed for {client_id}")
    
    async def _generate_response(self, client_id: str, user_text: str) -> None:
        """
        生成响应
        
        Args:
            client_id: 客户端 ID
            user_text: 用户输入文本
        """
        ctx = self._clients.get(client_id)
        if not ctx:
            return
        
        ctx.is_generating = True
        ctx.is_interrupted = False
        
        start_time = time.time()
        full_response = ""
        audio_sequence = 0
        
        try:
            # 发送 LLM 开始
            await self.ws_server.send_text(client_id, MessageType.LLM_START, "")
            
            # 创建文本分块器
            text_chunks = asyncio.Queue()
            
            async def on_text_chunk(chunk: TextChunk):
                """文本块回调"""
                await text_chunks.put(chunk)
            
            # 启动 TTS 处理任务
            tts_task = asyncio.create_task(
                self._process_tts(client_id, text_chunks)
            )
            
            # 流式生成 LLM 响应
            chunker = TextChunker(
                max_chunk_length=20,
                on_chunk_ready=lambda c: asyncio.create_task(on_text_chunk(c))
            )
            
            # 发送 TTS 开始
            await self.ws_server.send_text(client_id, MessageType.TTS_START, "")
            
            async for token in self.llm.generate_stream(ctx.conversation.get_messages()):
                if ctx.is_interrupted:
                    break
                
                full_response += token
                
                # 发送 token 给客户端（可选，用于显示）
                await self.ws_server.send_text(
                    client_id,
                    MessageType.LLM_TOKEN,
                    token
                )
                
                # 分块处理
                chunker.add_token(token)
            
            # 刷新剩余文本
            if not ctx.is_interrupted:
                chunker.flush()
            
            # 发送结束标记
            await text_chunks.put(None)
            
            # 等待 TTS 完成
            await tts_task
            
            # 发送 LLM 结束
            await self.ws_server.send_text(client_id, MessageType.LLM_END, "")
            
            # 发送 TTS 结束
            await self.ws_server.send_text(client_id, MessageType.TTS_END, "")
            
            # 保存助手响应到上下文
            if full_response and not ctx.is_interrupted:
                ctx.conversation.add_assistant_message(full_response)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"Response generated for {client_id}: {len(full_response)} chars in {total_time:.1f}ms")
            logger.info(f"LLM Response: {full_response}")
            
        except asyncio.CancelledError:
            logger.info(f"Generation cancelled for {client_id}")
        except Exception as e:
            logger.error(f"Generation error for {client_id}: {e}")
            await self.ws_server.send_text(
                client_id,
                MessageType.ERROR,
                str(e)
            )
        finally:
            ctx.is_generating = False
    
    async def _process_tts(
        self,
        client_id: str,
        text_queue: asyncio.Queue
    ) -> None:
        """
        处理 TTS 合成
        
        Args:
            client_id: 客户端 ID
            text_queue: 文本块队列
        """
        ctx = self._clients.get(client_id)
        audio_sequence = 0
        
        while True:
            try:
                chunk = await text_queue.get()
                
                if chunk is None:
                    break
                
                if ctx and ctx.is_interrupted:
                    break
                
                # 合成音频
                async for audio_chunk in self.tts.synthesize_stream(chunk.text):
                    if ctx and ctx.is_interrupted:
                        break
                    
                    # 发送音频块
                    await self.ws_server.send_audio(
                        client_id,
                        audio_chunk.pcm_data,
                        audio_sequence,
                        audio_chunk.sample_rate,
                        audio_chunk.is_last
                    )
                    audio_sequence += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS processing error: {e}")
                break
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "uptime": time.time() - self._start_time,
            "total_requests": self._total_requests,
            "connected_clients": self.ws_server.client_count,
            "client_ids": self.ws_server.client_ids
        }


async def run_server(args):
    """运行服务器"""
    server = VoiceAIServer(
        host=args.host,
        port=args.port,
        use_mock=args.mock
    )
    
    await server.start()
    
    print("\n" + "="*50)
    print("  VoiceAI 实时语音对话系统 - 云端服务")
    print("="*50)
    print(f"\n服务地址: ws://{args.host}:{args.port}")
    print(f"模式: {'模拟' if args.mock else '生产'}")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        while True:
            await asyncio.sleep(10)
            stats = server.get_stats()
            logger.debug(f"Stats: clients={stats['connected_clients']}, requests={stats['total_requests']}")
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        await server.stop()
        print("服务已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VoiceAI Cloud Server')
    parser.add_argument('--host', '-H', default='0.0.0.0',
                       help='Bind address')
    parser.add_argument('--port', '-p', type=int, default=NetworkConfig.WEBSOCKET_PORT,
                       help='Port number')
    parser.add_argument('--mock', '-m', action='store_true',
                       help='Use mock mode (no real models)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行服务器
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
