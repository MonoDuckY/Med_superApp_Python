@echo off
chcp 65001 > nul
echo === ĐANG CHẠY CHƯƠNG TRÌNH XỬ LÝ ẢNH ===

:: Kiểm tra xem môi trường ảo đã được tạo chưa
if not exist env\Scripts\activate (
    echo [LỖI] Chưa cài đặt môi trường! Vui lòng chạy file setup.bat trước.
    pause
    exit /b
)

:: Kích hoạt môi trường ảo và chạy code python
call env\Scripts\activate
python process_images.py

echo.
echo Nhấn phím bất kỳ để thoát.
pause