#!/usr/bin/env python3
"""
Simple USB Camera Live Feed Program
Displays live video from a USB webcam using OpenCV
Press 'q' or ESC to exit
"""

import cv2
import sys


def main():
    """Main function to capture and display camera feed."""
    # Initialize camera capture (0 is typically the default USB camera)
    print("Initializing camera...")
    camera = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        print("Make sure your USB camera is connected and not being used by another application.")
        sys.exit(1)
    
    # Set camera properties for better performance (optional)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera initialized successfully!")
    print("Press 'q' or ESC to exit...")
    
    # Create a window
    window_name = "USB Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()
            
            # Check if frame was successfully captured
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            # Display the resulting frame
            cv2.imshow(window_name, frame)
            
            # Wait for key press (1ms delay)
            # Break loop on 'q' or ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    
    finally:
        # Release camera and close all windows
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()

