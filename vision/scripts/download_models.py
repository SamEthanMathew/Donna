#!/usr/bin/env python3
"""
Download SCRFD and ArcFace ONNX models for face recognition.

This script attempts multiple download methods and provides clear instructions.
"""

import os
import sys
from pathlib import Path


def print_manual_instructions():
    """Print comprehensive manual download instructions."""
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "data" / "models"
    
    print("\n" + "="*70)
    print("📥 MANUAL MODEL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print()
    print("The face recognition system requires 2 ONNX models:")
    print()
    print("1. SCRFD 2.5G BNKPS (~3-4 MB) - Face detector with landmarks")
    print("2. ArcFace ResNet50 (~160-170 MB) - Face embedder")
    print()
    print("="*70)
    print("METHOD 1: Download via Python (Recommended)")
    print("="*70)
    print()
    print("Run these commands:")
    print()
    print("  cd", str(models_dir))
    print()
    print("  # Download SCRFD detector")
    print("  python3 << 'EOF'")
    print("import urllib.request")
    print("url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps-99a96e1c.onnx'")
    print("urllib.request.urlretrieve(url, 'scrfd_2.5g_bnkps.onnx')")
    print("print('✓ Downloaded scrfd_2.5g_bnkps.onnx')")
    print("EOF")
    print()
    print("  # Download ArcFace embedder")
    print("  python3 << 'EOF'")
    print("import urllib.request")
    print("url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'")
    print("urllib.request.urlretrieve(url, 'buffalo_l.zip')")
    print("import zipfile")
    print("with zipfile.ZipFile('buffalo_l.zip', 'r') as z:")
    print("    z.extract('w600k_r50.onnx', '.')")
    print("import os")
    print("os.rename('w600k_r50.onnx', 'arcface_r100_ms1mv3.onnx')")
    print("os.remove('buffalo_l.zip')")
    print("print('✓ Downloaded arcface_r100_ms1mv3.onnx')")
    print("EOF")
    print()
    print("="*70)
    print("METHOD 2: Using InsightFace Python package")
    print("="*70)
    print()
    print("Install and use the official InsightFace package:")
    print()
    print("  pip install insightface")
    print()
    print("  python3 << 'EOF'")
    print("import insightface")
    print("from insightface.app import FaceAnalysis")
    print("app = FaceAnalysis(providers=['CPUExecutionProvider'])")
    print("app.prepare(ctx_id=0, det_size=(640, 640))")
    print("# Models will be downloaded to ~/.insightface/models/")
    print("# Then copy them to:", str(models_dir))
    print("EOF")
    print()
    print("="*70)
    print("METHOD 3: Download from browser")
    print("="*70)
    print()
    print("Download these files and save to:", str(models_dir))
    print()
    print("Model 1 - SCRFD Detector:")
    print("  https://huggingface.co/deepinsight/insightface/resolve/main/models/scrfd_2.5g_bnkps.onnx")
    print("  Save as: scrfd_2.5g_bnkps.onnx")
    print()
    print("Model 2 - ArcFace Embedder:")
    print("  https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx")
    print("  Save as: arcface_r100_ms1mv3.onnx")
    print()
    print("  Note: You may need to create a free Hugging Face account")
    print()
    print("="*70)
    print()


def try_python_download():
    """Try downloading using Python's urllib."""
    import urllib.request
    import zipfile
    import tempfile
    
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Download SCRFD
    scrfd_path = models_dir / "scrfd_2.5g_bnkps.onnx"
    if not scrfd_path.exists():
        print("\n📥 Downloading SCRFD detector...")
        try:
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps-99a96e1c.onnx"
            urllib.request.urlretrieve(url, str(scrfd_path))
            if scrfd_path.exists() and scrfd_path.stat().st_size > 1_000_000:
                print(f"✓ Downloaded: {scrfd_path.name} ({scrfd_path.stat().st_size / 1024 / 1024:.1f} MB)")
                success_count += 1
            else:
                print("✗ Download failed or file too small")
                scrfd_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"✗ Error: {e}")
            scrfd_path.unlink(missing_ok=True)
    else:
        print(f"✓ SCRFD already exists: {scrfd_path.name}")
        success_count += 1
    
    # Download ArcFace
    arcface_path = models_dir / "arcface_r100_ms1mv3.onnx"
    if not arcface_path.exists():
        print("\n📥 Downloading ArcFace embedder (this may take a while, ~275 MB)...")
        try:
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                temp_zip = Path(tmp.name)
            
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    if block_num % 100 == 0:  # Update every 100 blocks
                        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f} / {mb_total:.1f} MB)", end='', flush=True)
            
            urllib.request.urlretrieve(url, str(temp_zip), reporthook=report_progress)
            print()  # New line after progress
            
            # Extract w600k_r50.onnx from the ZIP
            print("  Extracting model from ZIP...")
            with zipfile.ZipFile(temp_zip, 'r') as z:
                # List contents to find the right file
                files = z.namelist()
                target_file = None
                for f in files:
                    if 'w600k_r50.onnx' in f:
                        target_file = f
                        break
                
                if target_file:
                    with z.open(target_file) as source:
                        with open(arcface_path, 'wb') as dest:
                            dest.write(source.read())
                    print(f"✓ Extracted: {arcface_path.name} ({arcface_path.stat().st_size / 1024 / 1024:.1f} MB)")
                    success_count += 1
                else:
                    print("✗ Could not find w600k_r50.onnx in ZIP")
            
            temp_zip.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            arcface_path.unlink(missing_ok=True)
            temp_zip.unlink(missing_ok=True)
    else:
        print(f"✓ ArcFace already exists: {arcface_path.name}")
        success_count += 1
    
    return success_count


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "data" / "models"
    
    print("="*70)
    print("ONNX Model Downloader for Face Recognition System")
    print("="*70)
    
    try:
        success_count = try_python_download()
        
        print("\n" + "="*70)
        print(f"Download Summary: {success_count}/2 models ready")
        print("="*70)
        
        if success_count == 2:
            print("\n✅ All models downloaded successfully!")
            print(f"\nModels location: {models_dir}")
            scrfd_path = models_dir / "scrfd_2.5g_bnkps.onnx"
            arcface_path = models_dir / "arcface_r100_ms1mv3.onnx"
            print(f"  • {scrfd_path.name} ({scrfd_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  • {arcface_path.name} ({arcface_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print("\nYou can now run:")
            print("  python3 register_person.py   # Register faces")
            print("  python3 recognize_person.py  # Recognize faces")
            return 0
        else:
            print(f"\n⚠️  Only {success_count}/2 models downloaded.")
            print_manual_instructions()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print_manual_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
