@echo off

cd /D "%~dp0"

echo Setup venv environment
call env_venv.bat

:install_pip
if exist %DIR%\python\Scripts\pip.exe goto :install_python_packages
echo Install pip...

:install_python_packages
echo Install dependencies...
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install --upgrade -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple

echo Install finished.
pause
