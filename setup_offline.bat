cd /D "%~dp0"

rem set http_proxy=http://127.0.0.1:7890 & set https_proxy=http://127.0.0.1:7890

echo Setup offline environment
call env_offline.bat

:install_pip
if exist %DIR%\python\Scripts\pip.exe goto :install_python_packages
echo Install pip...
python %PIP_INSTALLER_LOCATION%

:install_python_packages
echo Install dependencies...
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple

echo Install finished.
pause
