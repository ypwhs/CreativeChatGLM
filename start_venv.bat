@echo off
cd /D "%~dp0"
if exist .venv goto :start

echo Setup venv
call setup_env.bat
goto :run

:start
echo Start venv
call .venv\Scripts\activate.bat
goto :run

:run
echo Start WebUI
python web_demo.py
pause
