@echo off

cd /D "%~dp0"

echo Start app.py
python app_fastapi.py %*

pause
