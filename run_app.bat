@echo off
title Smart Fridge Recommender - Finoria-co
cls
echo ========================================================
echo   Starting Smart Fridge Recommender (Finoria-co)
echo ========================================================
echo.
echo Starting Flask server and loading YOLO11s AI model...
echo Browser will open automatically at http://127.0.0.1:5000
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

echo.
echo Server stopped.
pause
