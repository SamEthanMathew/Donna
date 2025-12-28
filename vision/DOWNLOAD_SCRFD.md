# Download SCRFD Detector Model

## Current Status

✅ **ArcFace embedder**: Downloaded (166 MB)  
❌ **SCRFD detector**: Needs manual download (3-4 MB)

## Quick Instructions

The SCRFD detector model needs to be downloaded manually. Here are the easiest methods:

### Method 1: Direct Download Link (Easiest)

Download this file and save it as `scrfd_2.5g_bnkps.onnx`:

**Google Drive Link:**
https://drive.google.com/file/d/1nR3O_6YmBzKn1V9VdNaUPpGvL7Xd1ZAC/view

**OneDrive Link (alternative):**
https://onedrive.live.com/?authkey=%21AEeO3VQXmCDqGZo&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC

Save the file to:
```
/home/sam/vision/data/models/scrfd_2.5g_bnkps.onnx
```

### Method 2: Using wget with working mirror

```bash
cd /home/sam/vision/data/models

# Try this mirror
wget --no-check-certificate \
  https://www.dropbox.com/s/53ftnlarhyrpkg2/scrfd_2.5g_bnkps.onnx?dl=1 \
  -O scrfd_2.5g_bnkps.onnx
```

### Method 3: Using gdown (Google Drive downloader)

```bash
pip3 install gdown

cd /home/sam/vision/data/models

gdown 1nR3O_6YmBzKn1V9VdNaUPpGvL7Xd1ZAC -O scrfd_2.5g_bnkps.onnx
```

### Method 4: From another machine

If you have another computer with internet access:

1. Download from: https://github.com/deepinsight/insightface/releases
2. Transfer the file to Jetson using `scp`:
   ```bash
   scp scrfd_2.5g_bnkps.onnx sam@<jetson-ip>:/home/sam/vision/data/models/
   ```

## Verify Download

After downloading, verify the model:

```bash
cd /home/sam/vision
python3 test_setup.py
```

You should see:
```
✓ Test 4: ONNX Models
  ✓ SCRFD detector: 3.4 MB
  ✓ ArcFace embedder: 166.3 MB
```

## Then Start Using the System!

Once both models are downloaded:

```bash
# Register faces
python3 register_person.py
# Open: http://<jetson-ip>:5000

# Run recognition  
python3 recognize_person.py
# Open: http://<jetson-ip>:5001
```

## Alternative: Use a Different Detector

If you can't download SCRFD, you can modify the code to use OpenCV's built-in face detector (less accurate but works):

```python
# Edit register_person.py and recognize_person.py
# Replace SCRFD detector with OpenCV cascade
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

Contact for help if you need assistance!

