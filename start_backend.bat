@echo off
title Aegis-JKN ML/DL Backend API Server
echo ======================================================================
echo  Aegis-JKN ML/DL Connector Server (FastAPI + WebSocket)
echo ======================================================================

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%backend\venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment tidak ditemukan di: %VENV_PYTHON%
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%backend\src"
echo Menjalankan Uvicorn server di http://127.0.0.1:8000...
echo Tekan CTRL+C untuk menghentikan server.
"%VENV_PYTHON%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

pause
