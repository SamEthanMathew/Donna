#!/bin/bash
# ==== Piper TTS (Amy medium) Setup for Jetson AGX Orin / JetPack 5 ====
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Piper TTS in $SCRIPT_DIR"

# 1) Install audio player (aplay) + wget
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y alsa-utils wget

# 2) Download Piper aarch64 binary
echo "Downloading Piper binary..."
mkdir -p bin
cd bin
# Use the latest release (2023.11.14-2) which has aarch64 support
wget -O piper_linux_aarch64.tar.gz \
  https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
chmod +x piper/piper
cd ..

# 3) Download Amy (medium) model files
echo "Downloading Amy medium model files..."
mkdir -p models
cd models
wget -O en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true"
wget -O en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true"
cd ..

# 4) Create core directory for Python module
echo "Creating Python module structure..."
mkdir -p core

# 5) Quick CLI test: generate wav + play
echo "Testing installation..."
cd bin/piper
echo "Hey Sam. Piper Amy is running locally on your Jetson AGX Orin." | \
  ./piper --model "$SCRIPT_DIR/models/en_US-amy-medium.onnx" --output_file /tmp/amy.wav
aplay /tmp/amy.wav

cd "$SCRIPT_DIR"
echo ""
echo "✓ Piper TTS setup complete!"
echo "Test the Python module with: python3 -m core.tts_piper"

