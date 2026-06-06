@echo off
setlocal
cd /d %~dp0

echo Starting Document Intelligence Lab...
echo.
echo If your conda environment is not active, run this manually first:
echo   conda activate docintellab
echo.

streamlit run app\streamlit_app.py
pause
