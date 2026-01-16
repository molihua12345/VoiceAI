# VoiceAI 实时语音对话系统

高性能低延迟的本地-云端协同实时语音对话系统。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     VoiceAI 系统架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────┐    WebSocket    ┌───────────────┐ │
│  │   Local Client           │◄──────────────► │ Cloud Server  │ │
│  │   (Windows + RTX 3060)   │     JSON/Binary │ (Ubuntu +     │ │
│  │                          │                 │  RTX 3080 Ti) │ │
│  │  ┌─────────────────────┐ │                 │               │ │
│  │  │ Audio Capture       │ │                 │ ┌───────────┐ │ │
│  │  │ (sounddevice)       │ │                 │ │ LLM       │ │ │
│  │  └─────────┬───────────┘ │                 │ │ (vLLM +   │ │ │
│  │            ▼             │                 │ │  Qwen2)   │ │ │
│  │  ┌─────────────────────┐ │   ASR Text     │ └─────┬─────┘ │ │
│  │  │ VAD (Silero-VAD)    │ │ ─────────────► │       │       │ │
│  │  └─────────┬───────────┘ │                 │       ▼       │ │
│  │            ▼             │                 │ ┌───────────┐ │ │
│  │  ┌─────────────────────┐ │                 │ │ Text      │ │ │
│  │  │ ASR (SenseVoice)    │ │                 │ │ Chunker   │ │ │
│  │  └─────────────────────┘ │                 │ └─────┬─────┘ │ │
│  │                          │                 │       │       │ │
│  │  ┌─────────────────────┐ │   TTS Audio    │       ▼       │ │
│  │  │ Audio Player        │ │ ◄───────────── │ ┌───────────┐ │ │
│  │  │ (sounddevice)       │ │                 │ │ TTS       │ │ │
│  │  └─────────────────────┘ │                 │ │(GPT-SoVITS│ │ │
│  │                          │                 │ └───────────┘ │ │
│  │  ┌─────────────────────┐ │   INTERRUPT    │               │ │
│  │  │ Barge-In Controller │ │ ─────────────► │               │ │
│  │  └─────────────────────┘ │                 │               │ │
│  └──────────────────────────┘                 └───────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

**本地客户端 (Windows):**
```bash
# 仅安装本地客户端依赖
pip install -e ".[local]"

# 仅安装云端服务依赖  
pip install -e ".[cloud]"

# 安装所有依赖
pip install -e ".[all]"

# 安装开发依赖
pip install -e ".[dev]"
```

### 2. 启动服务

**方式一：模拟模式测试 (不需要下载模型)**
```bash
# Windows
test_mock.bat
```

**方式二：完整模式**

先在云端机器启动服务：
```bash
# Ubuntu
chmod +x start_server.sh
./start_server.sh
```

然后在本地启动客户端：
```bash
# Windows
start_local.bat -s ws://云端IP:8765
```

### 3. 开始对话

启动后，直接对着麦克风说话即可。系统会自动检测语音、转写、生成回复并播放。

## 命令行参数

### 本地客户端
```
python -m local_client.main [options]

选项:
  -s, --server URL    云端服务地址 (默认: ws://localhost:8765)
  --no-gpu            禁用 GPU 加速
  -d, --debug         启用调试日志
```

### 云端服务
```
python -m cloud_server.main [options]

选项:
  -H, --host HOST     绑定地址 (默认: 0.0.0.0)
  -p, --port PORT     端口号 (默认: 8765)
  -m, --mock          使用模拟模式 (不加载真实模型)
  -d, --debug         启用调试日志
```

## 项目结构

```
VoiceAI/
├── local_client/          # 本地客户端
│   ├── __init__.py
│   ├── main.py            # 客户端主程序
│   ├── audio_buffer.py    # 音频采集与播放
│   ├── vad_module.py      # 语音活动检测
│   ├── asr_engine.py      # 语音识别引擎
│   └── websocket_client.py# WebSocket 客户端
│
├── cloud_server/          # 云端服务
│   ├── __init__.py
│   ├── main.py            # 服务端主程序
│   ├── llm_engine.py      # LLM 推理引擎
│   ├── text_chunker.py    # 文本分块器
│   ├── tts_engine.py      # TTS 合成引擎
│   └── websocket_server.py# WebSocket 服务端
│
├── shared/                # 共享模块
│   ├── __init__.py
│   ├── protocol.py        # 通信协议定义
│   └── config.py          # 配置管理
│
├── config.yaml            # 配置文件
├── requirements_local.txt # 本地客户端依赖
├── requirements_cloud.txt # 云端服务依赖
├── start_local.bat        # 本地启动脚本 (Windows)
├── start_server.bat       # 服务端启动脚本 (Windows)
├── start_server.sh        # 服务端启动脚本 (Linux)
└── test_mock.bat          # 模拟模式测试脚本
```

## 核心特性

- **低延迟**: 端到端延迟控制在 1200ms 以内
- **流式处理**: LLM 和 TTS 均支持流式输出
- **打断机制**: 支持用户随时打断 AI 回复
- **全双工**: 同时支持语音输入和输出
- **GPU 加速**: 本地 ASR 和云端 LLM/TTS 均支持 GPU

## 技术栈

| 组件 | 技术 |
|------|------|
| VAD | Silero-VAD |
| ASR | SenseVoiceSmall (FunASR) |
| LLM | Qwen2-7B-AWQ (vLLM) |
| TTS | GPT-SoVITS v4 |
| 通信 | WebSocket |

## 配置说明

详细配置请参考 `config.yaml` 文件。

## 许可证

MIT License
