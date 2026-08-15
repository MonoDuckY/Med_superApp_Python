@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay Python environment .venv.
  pause
  exit /b 1
)

if not exist "input" mkdir input
if not exist "output" mkdir output
set "PYTHONPATH=%CD%\lama"
echo Quet anh trong thu muc input va xu ly bang LaMa...
echo.
".venv\Scripts\python.exe" process_batch.py
echo.
pause
