import cv2
import argparse
from vision.face_detector import YuNetFaceDetector

MODEL_PATH = "data/models/yunet.onnx"

def draw_face(frame, face_row):
    x, y, w, h = face_row[:4].astype(int)
    score = float(face_row[4])

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{score:.2f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

def main():
    parser = argparse.ArgumentParser(description="Run face detection on camera feed.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 0)")
    args = parser.parse_args()

    detector = YuNetFaceDetector(MODEL_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        # Draw all faces
        for face in faces:
            draw_face(frame, face)

        cv2.imshow("YuNet Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
