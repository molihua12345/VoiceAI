@echo off
REM VoiceAI 快速测试脚本 (模拟模式)
REM 在本地同时启动服务端和客户端进行测试

echo ========================================
echo   VoiceAI 快速测试 (模拟模式)
echo ========================================
echo.

REM 设置环境变量
set PYTHONPATH=%~dp0

echo [1/2] Starting mock server in background...
start "VoiceAI Server (Mock)" cmd /c "python -m cloud_server.main -m -d"

echo Waiting for server to start...
timeout /t 3 /nobreak > nul

echo.
echo [2/2] Starting local client...
echo.
python -m local_client.main -s ws://localhost:8765 -d

echo.
echo Test completed.
pause
