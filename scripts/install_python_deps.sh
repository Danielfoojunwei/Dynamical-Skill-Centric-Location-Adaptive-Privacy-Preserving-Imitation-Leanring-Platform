#!/bin/bash
# Install Python Dependencies for Dynamical Edge System

set -e

echo "============================================="
echo "Installing Python Dependencies..."
echo "============================================="

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing packages from requirements.txt..."
# Note: On Jetson, torch/torchvision might need to be installed from NVIDIA's index or pre-installed.
# We attempt standard install here, but if it fails, user might need to install manually.
pip install -r requirements.txt

echo "============================================="
echo "Python dependencies installed successfully!"
echo "To activate: source venv/bin/activate"
echo "============================================="
