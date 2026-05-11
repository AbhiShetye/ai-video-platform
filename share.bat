@echo off
title FrameAI - Share with Friends (Free)
color 0A

echo.
echo  ==========================================
echo   FrameAI - Free Public Sharing
echo   Files auto-deleted after 24h
echo  ==========================================
echo.

:: Start backend in background
echo  [1/3] Starting FrameAI server...
cd /d "%~dp0backend"
start /B python -m uvicorn main:app --host 127.0.0.1 --port 8000 > ..\share_server.log 2>&1
timeout /t 4 /nobreak >nul

:: Check if server started
curl -s http://127.0.0.1:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Server failed to start. Check share_server.log
    pause
    exit /b 1
)
echo  [OK] Server running at http://127.0.0.1:8000

:: Download cloudflared if not present
echo  [2/3] Setting up Cloudflare Tunnel...
if not exist "%~dp0cloudflared.exe" (
    echo  Downloading cloudflared (one-time, ~30MB)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%~dp0cloudflared.exe'"
    if %errorlevel% neq 0 (
        echo  [ERROR] Download failed. Check your internet connection.
        pause
        exit /b 1
    )
)

echo  [3/3] Creating public URL...
echo.
echo  ==========================================
echo   SHARE THIS LINK WITH YOUR FRIEND:
echo   (It will appear below in a moment...)
echo  ==========================================
echo.
echo  Your friend can open it directly in their
echo  browser - no login, no download needed.
echo.
echo  Files are automatically deleted after 24h.
echo.
echo  Press Ctrl+C to stop sharing.
echo  ==========================================
echo.

:: Start tunnel - URL appears in output
"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:8000

pause
