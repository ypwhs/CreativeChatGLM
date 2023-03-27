@echo off
cd /D "%~dp0"

call env_venv.bat

echo Start WebUI
python web_demo.py
pause
