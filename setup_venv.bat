cd /D "%~dp0"

echo Setup venv
python -m venv .venv
call .venv\Scripts\activate.bat

echo "Install dependencies"
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install --upgrade -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pause