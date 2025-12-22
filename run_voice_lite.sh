#!/bin/bash
# Wrapper script to run voice lite (text input, voice output) with proper environment setup

cd "$(dirname "$0")"
source .venv/bin/activate

# Fix for static TLS issue on Jetson
export LD_PRELOAD=.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

python app/voice_test_lite.py

