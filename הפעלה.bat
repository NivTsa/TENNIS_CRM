@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Tennis CRM Dashboard
echo   הדאשבורד ייפתח בדפדפן בכתובת:
echo   http://localhost:8501
echo   לעצירה: סגור את החלון הזה או Ctrl+C
echo ============================================
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
python -m streamlit run app.py
pause
