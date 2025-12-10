#!/bin/bash
set -e

echo "=== Installing Vision Dependencies (MMPose, RTMW3D) ==="

# 1. Install OpenMMLab dependencies using MIM
echo "Installing MIM and MMCV..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 2. Install MMPose
echo "Installing MMPose..."
mim install "mmpose>=1.0.0"

# 3. Install RTMW3D dependencies (if any specific ones besides mmpose)
# RTMW3D is usually a model config within MMPose, but sometimes requires extra libs
echo "Installing RTMW3D extras..."
pip install onnxruntime-gpu  # For ONNX inference if needed

echo "=== Vision Dependencies Installed Successfully ==="
