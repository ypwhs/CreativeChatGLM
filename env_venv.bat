@echo off

set DIR=.venv

cd /D "%~dp0"

if exist %DIR% goto :activate
echo Setup venv
python -m venv .venv

:activate
echo Activate venv
call .venv\Scripts\activate.bat
