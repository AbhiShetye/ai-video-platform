@echo off
title FrameAI Server
cd /d "%~dp0backend"
echo ============================================
echo  FrameAI Server
echo  Open browser: http://127.0.0.1:8000
echo  Keep this window open while using the app.
echo ============================================
echo.
:loop
python -m uvicorn main:app --host 127.0.0.1 --port 8000
echo.
echo Server stopped. Restarting in 2 seconds...
timeout /t 2 /nobreak >nul
goto loop
