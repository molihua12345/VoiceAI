#!/bin/bash
# VoiceAI 云端服务启动脚本 (Linux)

echo "========================================"
echo "  VoiceAI 实时语音对话系统 - 云端服务"
echo "========================================"
echo ""

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

# 默认参数
HOST="0.0.0.0"
PORT=8765
MOCK_MODE=""
DEBUG_MODE=""
NO_INTERRUPT=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--mock)
            MOCK_MODE="-m"
            shift
            ;;
        -d|--debug)
            DEBUG_MODE="-d"
            shift
            ;;
        -n|--no-interrupt)
            NO_INTERRUPT="-n"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Starting cloud server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Mock mode: ${MOCK_MODE:-disabled}"
echo "Interrupt: ${NO_INTERRUPT:+disabled}${NO_INTERRUPT:-enabled}"
echo ""

python3 -m cloud_server.main -H "$HOST" -p "$PORT" $MOCK_MODE $DEBUG_MODE $NO_INTERRUPT

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Server exited with error"
fi
