@echo off

cd /D "%~dp0"

call env_offline.bat

echo Start web_demo.py
python web_demo.py %*

pause
