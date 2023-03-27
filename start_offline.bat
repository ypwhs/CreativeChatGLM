@echo off

cd %~dp0

echo Setup offline environment
call offline_environment.bat

echo Start web_demo.py
python web_demo.py %*

pause
