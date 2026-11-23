@echo off
title Smart Fridge Recommender - Finoria-co
cls
echo ========================================================
echo   Starting Smart Fridge Recommender (Finoria-co)
echo ========================================================
echo.
echo Starting Flask server with YOLO11s AI model...
echo URL: http://127.0.0.1:5000
echo.

start "" "http://127.0.0.1:5000"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

pause
