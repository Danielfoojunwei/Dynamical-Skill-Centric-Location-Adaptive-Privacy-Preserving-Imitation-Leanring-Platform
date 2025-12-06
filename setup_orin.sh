#!/bin/bash
# Setup Script for Dynamical Edge on AGX Orin
set -e

echo "=================================================="
echo "   DYNAMICAL EDGE - SYSTEM SETUP"
echo "=================================================="

if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root (sudo ./setup_orin.sh)"
  exit 1
fi

echo "[1/4] Installing System Dependencies..."
apt update
apt install -y python3-pip python3-venv nodejs npm libgl1-mesa-glx

echo "[2/4] Configuring Permissions..."
# Add user to dialout group for USB Serial access (DYGlove)
# Assuming the script is run with sudo, we need the actual user
ACTUAL_USER=${SUDO_USER:-$USER}
echo "Adding user $ACTUAL_USER to dialout group..."
usermod -a -G dialout $ACTUAL_USER

echo "[3/4] Setting up Python Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# 1. Install OS Dependencies
echo "Installing OS dependencies..."
sudo ./scripts/install_os_deps.sh

# 2. Install Python Dependencies
echo "Installing Python dependencies..."
./scripts/install_python_deps.sh

# 3. Setup Environment
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || echo "API_KEY=default_insecure_key" > .env
fi

echo "Setup Complete! Run ./launch_orin.sh to start."
echo "=================================================="
echo "   SETUP COMPLETE"
echo "=================================================="
echo "Please log out and log back in for group permissions to take effect."
echo "Then run: ./launch_orin.sh"
