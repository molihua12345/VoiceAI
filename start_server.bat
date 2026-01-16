@echo off
REM VoiceAI 云端服务启动脚本 (Windows - 用于本地测试)

echo ========================================
echo   VoiceAI 实时语音对话系统 - 云端服务
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
set HOST=0.0.0.0
set PORT=8765
set MOCK_MODE=
set DEBUG_MODE=

:parse_args
if "%~1"=="" goto run
if "%~1"=="-H" (
    set HOST=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-p" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-m" (
    set MOCK_MODE=-m
    shift
    goto parse_args
)
if "%~1"=="--mock" (
    set MOCK_MODE=-m
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
shift
goto parse_args

:run
echo Starting cloud server...
echo Host: %HOST%
echo Port: %PORT%
echo.

python -m cloud_server.main -H %HOST% -p %PORT% %MOCK_MODE% %DEBUG_MODE%

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with error
    pause
)
