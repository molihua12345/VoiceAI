@echo off
REM VoiceAI 本地客户端启动脚本 (Windows)

echo ========================================
echo   VoiceAI 实时语音对话系统 - 本地客户端
echo ========================================
echo.

REM 检查 Python 环境
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM 设置环境变量
set PYTHONPATH=%~dp0
set CUDA_VISIBLE_DEVICES=0

REM 解析参数
set SERVER_URL=ws://117.50.193.188:8765
set DEBUG_MODE=
set NO_GPU=

:parse_args
if "%~1"=="" goto run
if "%~1"=="-s" (
    set SERVER_URL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--server" (
    set SERVER_URL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-d" (
    set DEBUG_MODE=-d
    shift
    goto parse_args
)
if "%~1"=="--debug" (
    set DEBUG_MODE=-d
    shift
    goto parse_args
)
if "%~1"=="--no-gpu" (
    set NO_GPU=--no-gpu
    shift
    goto parse_args
)
shift
goto parse_args

:run
echo Starting local client...
echo Server: %SERVER_URL%
echo.

python -m local_client.main -s %SERVER_URL% %DEBUG_MODE% %NO_GPU%

if errorlevel 1 (
    echo.
    echo [ERROR] Client exited with error
    pause
)
