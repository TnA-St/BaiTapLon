@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] Chuan bi moi truong Python...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc moi truong. Hay cai Python 3.10+ va thu lai.
        pause
        exit /b 1
    )
)
echo [2/3] Cai thu vien can thiet...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Loi cai thu vien. Hay kiem tra Internet va dung luong o dia.
    pause
    exit /b 1
)
echo [3/3] Mo giao dien Streamlit...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
