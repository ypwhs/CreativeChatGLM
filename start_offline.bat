@echo off

cd %~dp0

set PYTHON=python
set MAIN_SCRIPT=web_demo.py

%PYTHON% %MAIN_SCRIPT% %*
pause
