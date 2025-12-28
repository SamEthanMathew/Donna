#!/usr/bin/env python3
"""
Test script to verify the face recognition system setup.
Checks dependencies, models, and camera availability.
"""

import sys
from pathlib import Path

print("="*70)
print("Face Recognition System - Setup Verification")
print("="*70)
print()

# Test 1: Python version
print("✓ Test 1: Python Version")
print(f"  Python {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("  ⚠️  WARNING: Python 3.8+ recommended")
print()

# Test 2: Required modules
print("✓ Test 2: Required Modules")
required_modules = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'flask': 'Flask',
    'onnxruntime': 'onnxruntime',
    'PIL': 'Pillow',
    'requests': 'requests',
}

missing_modules = []
for module, package in required_modules.items():
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT INSTALLED")
        missing_modules.append(package)

if missing_modules:
    print(f"\n  ⚠️  Missing modules: {', '.join(missing_modules)}")
    print("  Run: pip install -r requirements.txt")
else:
    print("  ✓ All required modules installed")
print()

# Test 3: ONNX Runtime providers
print("✓ Test 3: ONNX Runtime Providers")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    for provider in providers:
        print(f"  • {provider}")
    
    if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
        print("  ✓ GPU acceleration available!")
    else:
        print("  ℹ️  CPU-only mode (GPU support not available)")
except Exception as e:
    print(f"  ✗ Error: {e}")
print()

# Test 4: ONNX Models
print("✓ Test 4: ONNX Models")
models_dir = Path(__file__).parent / "data" / "models"
scrfd_path = models_dir / "scrfd_2.5g_bnkps.onnx"
arcface_path = models_dir / "arcface_r100_ms1mv3.onnx"

models_ok = True
if scrfd_path.exists():
    size_mb = scrfd_path.stat().st_size / (1024*1024)
    print(f"  ✓ SCRFD detector: {size_mb:.1f} MB")
else:
    print(f"  ✗ SCRFD detector: NOT FOUND")
    models_ok = False

if arcface_path.exists():
    size_mb = arcface_path.stat().st_size / (1024*1024)
    print(f"  ✓ ArcFace embedder: {size_mb:.1f} MB")
else:
    print(f"  ✗ ArcFace embedder: NOT FOUND")
    models_ok = False

if not models_ok:
    print("\n  ⚠️  Run: python scripts/download_models.py")
print()

# Test 5: Camera
print("✓ Test 5: Camera Access")
try:
    import cv2
    camera = cv2.VideoCapture(0)
    
    if camera.isOpened():
        ret, frame = camera.read()
        if ret and frame is not None:
            print(f"  ✓ Camera accessible: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("  ⚠️  Camera opened but cannot read frames")
        camera.release()
    else:
        print("  ✗ Cannot open camera")
        print("  Check: ls /dev/video*")
except Exception as e:
    print(f"  ✗ Error: {e}")
print()

# Test 6: Directory structure
print("✓ Test 6: Directory Structure")
required_dirs = [
    "data/models",
    "data/db",
    "data/faces",
    "scripts",
    "vision",
]

dirs_ok = True
for dir_path in required_dirs:
    full_path = Path(__file__).parent / dir_path
    if full_path.exists():
        print(f"  ✓ {dir_path}/")
    else:
        print(f"  ✗ {dir_path}/ - NOT FOUND")
        dirs_ok = False

if not dirs_ok:
    print("\n  ⚠️  Some directories missing. Re-run setup?")
print()

# Test 7: Vision module import
print("✓ Test 7: Vision Module")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from vision.face_store import FaceStore
    from vision.face_align import align_face_112
    print("  ✓ face_store")
    print("  ✓ face_align")
    
    if models_ok:
        from vision.scrfd_detector import SCRFDFaceDetector
        from vision.arcface_embedder import ArcFaceEmbedder
        print("  ✓ scrfd_detector")
        print("  ✓ arcface_embedder")
except Exception as e:
    print(f"  ✗ Error importing: {e}")
print()

# Summary
print("="*70)
if missing_modules or not models_ok or not dirs_ok:
    print("⚠️  Setup incomplete - please address issues above")
    print("="*70)
    sys.exit(1)
else:
    print("✓ Setup verification passed!")
    print("="*70)
    print()
    print("You're ready to use the face recognition system!")
    print()
    print("Quick start:")
    print("  1. Register faces:   python3 register_person.py")
    print("  2. Run recognition:  python3 recognize_person.py")
    print()
    sys.exit(0)

