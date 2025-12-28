#!/bin/bash
# Quick setup script for Face Recognition System

echo "========================================================================"
echo "Face Recognition System - Setup"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if user wants GPU support
echo ""
read -p "Do you want to install ONNX Runtime GPU support? (recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing ONNX Runtime GPU..."
    pip install --extra-index-url https://pypi.nvidia.com onnxruntime-gpu
fi

# Download models
echo ""
echo "Downloading ONNX models..."
python3 scripts/download_models.py

# Check setup
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Register faces:   python3 register_person.py"
echo "  2. Run recognition:  python3 recognize_person.py"
echo ""
echo "Then open in your browser:"
echo "  - Registration:  http://$(hostname -I | awk '{print $1}'):5000"
echo "  - Recognition:   http://$(hostname -I | awk '{print $1}'):5001"
echo ""
echo "See README.md for detailed documentation."
echo "========================================================================"

