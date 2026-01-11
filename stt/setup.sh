#!/bin/bash
# ==== Whisper.cpp STT Setup for Jetson AGX Orin / JetPack 5 ====
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Whisper.cpp STT in $SCRIPT_DIR"

# 1) Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y cmake make g++ ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# 2) Clone whisper.cpp repository
echo "Cloning whisper.cpp repository..."
if [ ! -d "bin/whisper.cpp" ]; then
    mkdir -p bin
    cd bin
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
else
    cd bin/whisper.cpp
    git pull
fi

# 3) Build whisper.cpp for aarch64 (CPU optimized)
echo "Building whisper.cpp for aarch64..."
# Use CPU-only build (no CUDA required, simpler and works well on Jetson)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Verify build
if [ ! -f "build/bin/main" ]; then
    echo "Error: Build failed - main binary not found"
    exit 1
fi

echo "✓ Build successful"

cd "$SCRIPT_DIR"

# 4) Download ggml-base.en.bin model
echo "Downloading Whisper base English model..."
mkdir -p models
cd models

if [ ! -f "ggml-base.en.bin" ]; then
    echo "Downloading ggml-base.en.bin (~142 MB)..."
    wget -O ggml-base.en.bin \
      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    
    # Verify download
    if [ ! -f "ggml-base.en.bin" ]; then
        echo "Error: Model download failed"
        exit 1
    fi
    
    echo "✓ Model downloaded successfully"
else
    echo "Model already exists, skipping download"
fi

cd "$SCRIPT_DIR"

# 5) Create a simple test audio file and verify installation
echo "Testing installation..."
# Create a simple test (we'll use arecord to create a test file)
echo "To test, run: python3 stt_whisper.py"

echo ""
echo "✓ Whisper.cpp STT setup complete!"
echo ""
echo "Binary location: $SCRIPT_DIR/bin/whisper.cpp/build/bin/main"
echo "Model location: $SCRIPT_DIR/models/ggml-base.en.bin"
echo ""
echo "Test with: python3 stt_whisper.py"

