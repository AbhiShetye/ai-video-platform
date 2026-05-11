@echo off
title FrameAI Server
color 0B
echo.
echo  Starting FrameAI...
echo  Open http://127.0.0.1:8000 in your browser
echo  Press Ctrl+C to stop
echo.
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
