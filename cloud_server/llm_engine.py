"""
LLM 推理引擎
基于 vLLM + Qwen2-7B-AWQ 实现高效流式文本生成
"""

from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = "Qwen/Qwen2-7B-Instruct-AWQ"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    gpu_memory_utilization: float = 0.75


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationContext:
    """
    对话上下文管理
    管理多轮对话历史
    """
    
    def __init__(self, max_history: int = 10, system_prompt: str = ""):
        """
        初始化对话上下文
        
        Args:
            max_history: 最大历史记录数
            system_prompt: 系统提示词
        """
        self.max_history = max_history
        self.system_prompt = system_prompt or "你是一个友好的AI助手，请用简洁自然的语言回答问题。"
        self.history: List[ConversationMessage] = []
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.history.append(ConversationMessage(role="user", content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.history.append(ConversationMessage(role="assistant", content=content))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """修剪历史记录"""
        if len(self.history) > self.max_history * 2:
            # 保留最近的对话
            self.history = self.history[-self.max_history * 2:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """获取消息列表（用于 LLM 输入）"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages
    
    def clear(self) -> None:
        """清空历史"""
        self.history = []


class LLMEngine:
    """
    LLM 推理引擎
    使用 vLLM 进行高效推理
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 引擎
        
        Args:
            config: LLM 配置
        """
        self.config = config or LLMConfig()
        self._llm = None
        self._tokenizer = None
        self._is_generating = False
        self._should_stop = False
        
        logger.info(f"LLM Engine initializing with model: {self.config.model_name}")
    
    def load_model(self) -> None:
        """加载模型"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            logger.info(f"Loading LLM model: {self.config.model_name}")
            start_time = time.time()
            
            # 加载 vLLM
            self._llm = LLM(
                model=self.config.model_name,
                trust_remote_code=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype="half",  # FP16
                quantization="awq",
                max_model_len=4096,
                enforce_eager=False,  # 启用 CUDA graph
            )
            
            # 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"LLM model loaded in {load_time:.1f}s")
            
        except ImportError as e:
            logger.error("vLLM not installed. Please install: pip install vllm")
            raise RuntimeError(f"LLM dependency missing: {e}")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise RuntimeError(f"LLM model loading failed: {e}")
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建提示词"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # 使用 tokenizer 的 apply_chat_template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本
        
        Args:
            messages: 消息列表
            on_token: token 回调
        
        Yields:
            生成的 token
        """
        if self._llm is None:
            raise RuntimeError("LLM not loaded")
        
        from vllm import SamplingParams
        
        self._is_generating = True
        self._should_stop = False
        
        try:
            # 构建提示词
            prompt = self._build_prompt(messages)
            
            # 采样参数
            sampling_params = SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            
            # 记录 TTFT
            start_time = time.time()
            first_token_time = None
            
            # 流式生成
            outputs = self._llm.generate(
                [prompt],
                sampling_params,
                use_tqdm=False
            )
            
            # vLLM 的流式输出
            generated_text = ""
            for output in outputs:
                if self._should_stop:
                    logger.info("Generation interrupted")
                    break
                
                for token_output in output.outputs:
                    new_text = token_output.text[len(generated_text):]
                    generated_text = token_output.text
                    
                    if new_text:
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = (first_token_time - start_time) * 1000
                            logger.debug(f"TTFT: {ttft:.1f}ms")
                        
                        if on_token:
                            on_token(new_text)
                        
                        yield new_text
                        
                        # 让出控制权
                        await asyncio.sleep(0)
                        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
        finally:
            self._is_generating = False
    
    def generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """
        同步生成文本
        
        Args:
            messages: 消息列表
        
        Returns:
            生成的文本
        """
        if self._llm is None:
            raise RuntimeError("LLM not loaded")
        
        from vllm import SamplingParams
        
        prompt = self._build_prompt(messages)
        
        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        outputs = self._llm.generate([prompt], sampling_params)
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""
    
    def interrupt(self) -> None:
        """中断当前生成"""
        if self._is_generating:
            self._should_stop = True
            logger.info("Generation interrupt requested")
    
    @property
    def is_generating(self) -> bool:
        """是否正在生成"""
        return self._is_generating


class MockLLMEngine:
    """
    模拟 LLM 引擎
    用于测试（不需要实际加载模型）
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._is_generating = False
        self._should_stop = False
        logger.info("MockLLMEngine initialized (for testing)")
    
    def load_model(self) -> None:
        """模拟加载模型"""
        logger.info("Mock model loaded")
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成模拟响应"""
        self._is_generating = True
        self._should_stop = False
        
        # 获取用户最后一条消息
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # 生成模拟响应
        response = f"您好！我收到了您的消息：'{user_message}'。这是一个测试响应，用于验证系统的流式输出功能。"
        
        try:
            # 模拟逐字输出
            for char in response:
                if self._should_stop:
                    break
                
                if on_token:
                    on_token(char)
                
                yield char
                await asyncio.sleep(0.03)  # 模拟生成延迟
                
        finally:
            self._is_generating = False
    
    def interrupt(self) -> None:
        """中断生成"""
        self._should_stop = True
    
    @property
    def is_generating(self) -> bool:
        return self._is_generating


def create_llm_engine(use_mock: bool = False, config: Optional[LLMConfig] = None):
    """
    创建 LLM 引擎
    
    Args:
        use_mock: 是否使用模拟引擎
        config: LLM 配置
    
    Returns:
        LLM 引擎实例
    """
    if use_mock:
        return MockLLMEngine(config)
    return LLMEngine(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    async def test_mock_llm():
        """测试模拟 LLM"""
        llm = MockLLMEngine()
        llm.load_model()
        
        context = ConversationContext()
        context.add_user_message("你好，请介绍一下你自己")
        
        print("Testing mock LLM generation:")
        full_response = ""
        
        async for token in llm.generate_stream(context.get_messages()):
            print(token, end='', flush=True)
            full_response += token
        
        print("\n\nFull response:", full_response)
    
    asyncio.run(test_mock_llm())
