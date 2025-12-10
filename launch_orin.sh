#!/bin/bash
# Launcher for Dynamical Edge on AGX Orin

echo "=================================================="
echo "   DYNAMICAL EDGE - AGX ORIN LAUNCHER"
echo "=================================================="

# 1. Activate Environment (assuming venv or conda)
source venv/bin/activate

# 2. Check for .env
if [ ! -f .env ]; then
    echo "[ERROR] .env file missing!"
    exit 1
fi

# 3. Start Backend
echo "[INFO] Starting Backend..."
cd src/platform/api
./../../venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../../../backend.log 2>&1 &
BACKEND_PID=$!
echo "[INFO] Backend PID: $BACKEND_PID"
cd ../../..

# 4. Start Frontend (assuming built)
# echo "[INFO] Starting Frontend..."
# cd ../ui
# npm run preview -- --host > ../../frontend.log 2>&1 &
# FRONTEND_PID=$!

echo "[INFO] System Running."
echo "Logs: backend.log"
echo "Press Ctrl+C to stop."

trap "kill $BACKEND_PID; exit" INT
wait
