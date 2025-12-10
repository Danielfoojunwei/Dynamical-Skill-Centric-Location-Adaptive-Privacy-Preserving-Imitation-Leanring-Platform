#!/bin/bash
# Install OS Dependencies for Dynamical Edge System (Ubuntu/Jetson)

set -e

echo "============================================="
echo "Installing OS Dependencies..."
echo "============================================="

if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root (sudo)"
  exit 1
fi

# Update apt
apt-get update

# Install utilities
apt-get install -y curl git build-essential

# Install Media/Vision libs
# Check if installed to avoid re-installing unnecessarily (though apt handles this)
apt-get install -y \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libssl-dev \
    libffi-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

echo "============================================="
echo "OS dependencies installed successfully!"
echo "============================================="
