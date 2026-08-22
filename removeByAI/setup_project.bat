@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==============================================
echo   Caliper Cleanroom - Project Setup
echo ==============================================

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Chua cai Python Launcher. Can Python 3.10.
  pause
  exit /b 1
)

py -3.10 --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Khong tim thay Python 3.10.
  echo Hay cai Python 3.10 va chay lai file nay.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Tao virtual environment Python 3.10...
  py -3.10 -m venv .venv
)

echo [2/5] Cap nhat pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [3/5] Cai backend va dependency co ban...
".venv\Scripts\python.exe" -m pip install -r requirements-lama.txt
if errorlevel 1 goto :failed

echo [4/5] Cai dependency LaMa tuong thich Python 3.10...
".venv\Scripts\python.exe" -m pip install pyyaml tqdm easydict scikit-image scikit-learn opencv-python joblib matplotlib pandas "albumentations==0.5.2" "imgaug==0.4.0" hydra-core pytorch-lightning tabulate "kornia==0.5.0" webdataset packaging lpips
if errorlevel 1 goto :failed

echo [5/5] Cai PyTorch CUDA 12.6 cho NVIDIA GPU...
".venv\Scripts\python.exe" -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :failed

echo Chinh NumPy de tuong thich voi Imgaug/LaMa...
".venv\Scripts\python.exe" -m pip install --no-deps "numpy==1.23.5"
if errorlevel 1 goto :failed

if not exist "big-lama\models\best.ckpt" (
  echo Dang tai model Big-LaMa, co the mat vai phut...
  if not exist "big-lama.zip" powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip' -OutFile 'big-lama.zip'"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'big-lama.zip' -DestinationPath '.' -Force"
)

set "PYTHONPATH=%CD%\lama"
echo.
echo Kiem tra GPU va model...
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
".venv\Scripts\python.exe" -c "from server import health; print('LaMa:', health())"
echo.
echo [DONE] Cai dat hoan tat. Chay start_backend.bat de mo app backend.
pause
exit /b 0

:failed
echo.
echo [ERROR] Cai dat that bai. Kiem tra log phia tren va chay lai.
pause
exit /b 1
