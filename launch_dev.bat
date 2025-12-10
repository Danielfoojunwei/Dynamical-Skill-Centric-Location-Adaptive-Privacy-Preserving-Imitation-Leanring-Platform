@echo off
echo ==================================================
echo    DYNAMICAL EDGE - DEV LAUNCHER
echo ==================================================

if not exist .env (
    echo [ERROR] .env file missing!
    pause
    exit /b
)

echo [INFO] Starting Backend...
start "Dynamical Backend" cmd /k "cd edge_platform\api && uvicorn main:app --reload"

echo [INFO] Starting Frontend...
start "Dynamical Frontend" cmd /k "cd edge_platform\ui && npm run dev"

echo [INFO] System Launched.
