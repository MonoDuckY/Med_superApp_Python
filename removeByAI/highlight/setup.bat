@echo off
chcp 65001 > nul
echo === BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG ===

:: 1. Tạo môi trường ảo venv tên là "env"
echo 1. Đang khởi tạo Virtual Environment (env)...
python -m venv env

:: 2. Kích hoạt môi trường ảo và cài đặt thư viện
echo 2. Đang cài đặt các thư viện cần thiết (opencv, numpy)...
call env\Scripts\activate
python -m pip install --upgrade pip
pip install opencv-python numpy

echo.
echo === CÀI ĐẶT HOÀN TẤT ===
echo Hãy tạo thư mục "input" (nếu chưa có), bỏ ảnh vào và chạy file run.bat.
pause