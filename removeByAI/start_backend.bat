@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv. Hay tao virtual environment truoc.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%\lama"
echo Dang khoi dong Caliper Cleanroom / LaMa tai http://127.0.0.1:8000
".venv\Scripts\python.exe" server.py
pause
