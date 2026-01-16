# Requirements Document

## Introduction

本文档定义了一套高性能低延迟的本地-云端协同实时语音对话系统的需求规范。该系统采用"边缘-中心协同架构"，将实时性要求高的感知层任务（音频采集、VAD、STT）部署在本地 Windows 11 + RTX 3060 环境，将计算密集型的认知层与合成层任务（LLM、TTS）部署在云端 Ubuntu + RTX 3080 Ti 环境。目标是将全链路延迟控制在 1 秒以内，远低于 10 秒的设计目标。

## Glossary

- **Local_Client**: 本地边缘端应用，运行于 Windows 11 + RTX 3060，负责音频采集、VAD、ASR 和音频播放
- **Cloud_Server**: 云端算力端服务，运行于 Ubuntu 22.04 + RTX 3080 Ti，负责 LLM 推理和 TTS 合成
- **VAD_Module**: 语音活动检测模块，基于 Silero-VAD 实现，用于检测用户是否正在说话
- **ASR_Engine**: 自动语音识别引擎，基于 SenseVoiceSmall 实现本地流式转写
- **LLM_Engine**: 大语言模型推理引擎，基于 vLLM + Qwen2-7B-AWQ 实现流式文本生成
- **TTS_Engine**: 文本转语音合成引擎，基于 GPT-SoVITS v4 实现高保真语音合成
- **Audio_Buffer**: 音频缓冲区，用于管理音频数据的采集和播放队列
- **WebSocket_Channel**: 基于 WebSocket 的双向通信通道，用于本地与云端的实时数据传输
- **Micro_Chunk**: 微块，LLM 输出按标点符号切分的文本片段，用于触发 TTS 合成
- **Barge_In**: 打断机制，允许用户在 AI 发言时随时打断并开始新的对话

## Requirements

### Requirement 1: 本地音频采集与管理

**User Story:** As a user, I want the system to capture my voice input with minimal latency, so that my speech can be processed in real-time.

#### Acceptance Criteria

1. WHEN the Local_Client starts, THE Audio_Buffer SHALL initialize with a 32ms frame size and 1024 samples per buffer
2. WHEN audio is being captured, THE Local_Client SHALL use non-blocking callback mode to avoid playback interruptions
3. WHEN audio capture begins, THE Audio_Buffer SHALL maintain a 100ms ring buffer to preserve speech onset
4. WHILE audio is being captured, THE Local_Client SHALL support simultaneous playback without blocking
5. IF audio driver initialization fails, THEN THE Local_Client SHALL report the error and suggest WASAPI/ASIO configuration

### Requirement 2: 语音活动检测 (VAD)

**User Story:** As a user, I want the system to accurately detect when I start and stop speaking, so that only relevant audio is processed.

#### Acceptance Criteria

1. WHEN audio frames are received, THE VAD_Module SHALL analyze each 32ms chunk for speech presence
2. WHEN speech is detected (is_speech_start), THE VAD_Module SHALL trigger ASR processing with the pre-buffered 100ms audio
3. WHEN speech ends (is_speech_end), THE VAD_Module SHALL signal the end of the current utterance
4. WHILE the system is playing audio, THE VAD_Module SHALL continue monitoring for user speech (full-duplex)
5. WHEN user speech is detected during playback, THE VAD_Module SHALL trigger the Barge_In mechanism

### Requirement 3: 本地语音识别 (ASR)

**User Story:** As a user, I want my speech to be transcribed quickly and accurately on my local device, so that the system can understand what I'm saying without cloud latency.

#### Acceptance Criteria

1. WHEN the Local_Client starts, THE ASR_Engine SHALL load SenseVoiceSmall model in FP16/ONNX format
2. WHEN speech audio is received, THE ASR_Engine SHALL transcribe it within 70ms for 10-second audio segments
3. WHEN partial transcription is available, THE ASR_Engine SHALL send incremental text to the cloud immediately
4. WHEN transcription completes, THE ASR_Engine SHALL output the final text with punctuation
5. IF ASR model loading fails, THEN THE Local_Client SHALL report GPU memory status and suggest optimization

### Requirement 4: 网络通信层

**User Story:** As a developer, I want secure and low-latency communication between local and cloud components, so that the system can maintain real-time responsiveness.

#### Acceptance Criteria

1. WHEN the system initializes, THE WebSocket_Channel SHALL establish a P2P connection via Tailscale
2. WHEN text is sent to cloud, THE WebSocket_Channel SHALL use JSON format with message type headers
3. WHEN audio is transmitted, THE WebSocket_Channel SHALL use binary frames without Base64 encoding
4. WHEN sending data, THE WebSocket_Channel SHALL disable Nagle algorithm (TCP_NODELAY) for minimal latency
5. IF connection is lost, THEN THE WebSocket_Channel SHALL attempt automatic reconnection with exponential backoff
6. WHEN network RTT exceeds 100ms, THE WebSocket_Channel SHALL log a warning for diagnostics

### Requirement 5: 云端 LLM 推理

**User Story:** As a user, I want intelligent and contextual responses to my queries, so that I can have meaningful conversations with the AI.

#### Acceptance Criteria

1. WHEN the Cloud_Server starts, THE LLM_Engine SHALL load Qwen2-7B-AWQ model via vLLM with PagedAttention
2. WHEN text input is received, THE LLM_Engine SHALL begin streaming token generation within 200ms (TTFT)
3. WHILE generating tokens, THE LLM_Engine SHALL maintain conversation context via KV cache
4. WHEN tokens are generated, THE LLM_Engine SHALL stream them to the text chunking module immediately
5. IF an INTERRUPT signal is received, THEN THE LLM_Engine SHALL terminate current generation and clear state

### Requirement 6: 文本微块化处理

**User Story:** As a developer, I want LLM output to be chunked intelligently, so that TTS can start synthesizing audio as early as possible.

#### Acceptance Criteria

1. WHEN tokens are received from LLM, THE Cloud_Server SHALL buffer them until a punctuation mark is detected
2. WHEN punctuation [,.\!?，。！？\n] is detected, THE Cloud_Server SHALL immediately send the accumulated text to TTS
3. WHEN accumulated text exceeds 20 characters without punctuation, THE Cloud_Server SHALL force a chunk split
4. WHEN a chunk is ready, THE Cloud_Server SHALL include sequence number for ordered playback

### Requirement 7: 云端语音合成 (TTS)

**User Story:** As a user, I want natural and high-quality voice responses, so that the conversation feels human-like.

#### Acceptance Criteria

1. WHEN the Cloud_Server starts, THE TTS_Engine SHALL load GPT-SoVITS v4 model with streaming_mode=3
2. WHEN a text chunk is received, THE TTS_Engine SHALL begin audio synthesis within 150ms
3. WHEN audio is synthesized, THE TTS_Engine SHALL output 48kHz PCM audio in streaming chunks
4. WHILE synthesizing, THE TTS_Engine SHALL use overlap_length=2 to prevent audio artifacts at chunk boundaries
5. IF an INTERRUPT signal is received, THEN THE TTS_Engine SHALL flush the synthesis queue immediately

### Requirement 8: 本地音频播放

**User Story:** As a user, I want to hear the AI's response smoothly without stuttering, so that the conversation feels natural.

#### Acceptance Criteria

1. WHEN audio chunks are received, THE Local_Client SHALL queue them in a thread-safe playback buffer
2. WHEN playback begins, THE Local_Client SHALL use callback-based streaming to prevent blocking
3. WHILE playing audio, THE Local_Client SHALL maintain buffer levels to handle network jitter
4. IF playback buffer underruns, THEN THE Local_Client SHALL insert silence and log the event
5. WHEN Barge_In is triggered, THE Local_Client SHALL immediately clear the playback buffer

### Requirement 9: 打断机制 (Barge-In)

**User Story:** As a user, I want to interrupt the AI at any time during its response, so that I can redirect the conversation naturally.

#### Acceptance Criteria

1. WHILE audio is playing, THE VAD_Module SHALL continue monitoring microphone input
2. WHEN user speech is detected with more than 2 characters transcribed, THE Local_Client SHALL send INTERRUPT signal
3. WHEN INTERRUPT is sent, THE Local_Client SHALL immediately stop audio playback and clear buffers
4. WHEN Cloud_Server receives INTERRUPT, THE LLM_Engine SHALL terminate generation and THE TTS_Engine SHALL flush queue
5. WHEN INTERRUPT completes, THE system SHALL be ready to process the new user input within 100ms

### Requirement 10: 系统性能与延迟

**User Story:** As a user, I want the system to respond quickly, so that the conversation feels natural and responsive.

#### Acceptance Criteria

1. THE system SHALL achieve end-to-end latency under 1200ms in normal conditions
2. THE system SHALL achieve end-to-end latency under 600ms in optimal conditions
3. WHEN processing, THE Local_Client SHALL utilize GPU acceleration for ASR inference
4. WHEN processing, THE Cloud_Server SHALL utilize GPU acceleration for LLM and TTS inference
5. THE system SHALL log latency metrics for each pipeline stage for performance monitoring

### Requirement 11: 错误处理与恢复

**User Story:** As a user, I want the system to handle errors gracefully, so that my conversation experience is not disrupted.

#### Acceptance Criteria

1. IF GPU memory is exhausted, THEN THE system SHALL log the error and attempt to free cached resources
2. IF WebSocket connection fails, THEN THE Local_Client SHALL display connection status and retry
3. IF ASR produces empty output, THEN THE system SHALL ignore the segment and continue listening
4. IF TTS synthesis fails, THEN THE Cloud_Server SHALL return an error message and continue processing
5. WHEN any component fails, THE system SHALL maintain partial functionality where possible
