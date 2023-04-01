@echo off

cd /D "%~dp0"

echo Start app.py
python app.py %*

pause
