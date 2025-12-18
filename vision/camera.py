import cv2

def main():
    # Open the camera explicitly via V4L2 (best for Jetson)
    cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

    if not cap.isOpened():
        print("ERROR: Could not open /dev/video1")
        return

    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break

        cv2.imshow("Jetson Camera Feed", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
