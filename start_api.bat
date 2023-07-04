@echo off

cd /D "%~dp0"

echo Start app_fastapi.py
python app_fastapi.py %*

pause
