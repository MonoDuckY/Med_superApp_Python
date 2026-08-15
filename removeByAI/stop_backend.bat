@echo off
setlocal
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do set "PORT_PID=%%P"

if defined PORT_PID (
  echo Dang tat backend PID %PORT_PID%...
  taskkill /PID %PORT_PID% /F
) else (
  echo Khong tim thay backend dang nghe tren port 8000.
)
pause
