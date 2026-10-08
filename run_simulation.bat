@echo off
title Aegis-JKN ML/DL & OASIS Simulation Runner
echo ======================================================================
echo  Aegis-JKN ML/DL & OASIS Simulation Runner (Windows)
echo ======================================================================

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%backend\venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment tidak ditemukan di: %VENV_PYTHON%
    echo Jalankan setup venv terlebih dahulu.
    pause
    exit /b 1
)

echo Menjalankan simulasi MHGSL + OASIS...
"%VENV_PYTHON%" "%SCRIPT_DIR%run_windows_simulation.py"

echo.
pause
