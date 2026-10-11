# File: yolo_real_time_detection.py

import cv2
import torch
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8 model
    model = YOLO('yolov5n.pt')  # Replace with yolov8s.pt or yolov8l.pt for higher accuracy

    # Open webcam for real-time video capture
    cap = cv2.VideoCapture(0)  # Change index if using an external camera

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to exit the video stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize detections
        annotated_frame = results[0].plot()  # Automatically draws boxes and labels

        # Display the processed frame
        cv2.imshow("YOLO Real-Time Object Detection", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
