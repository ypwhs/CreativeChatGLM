cd /D "%~dp0"

echo "Setup VENV"
call environment.bat

python %PIP_INSTALLER_LOCATION%

echo "Install dependencies"
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install --upgrade -r CreativeChatGLM/requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple

pause
