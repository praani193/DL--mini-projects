import cv2
import numpy as np

# ESP32-CAM URL (adjust based on your setup)
url = "http://<ESP32_CAM_IP>:<PORT>/video"  # Replace <ESP32_CAM_IP> and <PORT> with your ESP32-CAM IP and port

# Start capturing video from ESP32-CAM stream
cap = cv2.VideoCapture(url)

# Check if video stream is available
if not cap.isOpened():
    print("Error: Unable to open video stream from ESP32-CAM")
    exit()

# Initialize variables for trajectory tracking
prev_frame = None
trajectory_points = []


# Function to process frames and detect motion
def detect_motion(frame, prev_frame):
    # Convert the current frame and previous frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and previous frame
    diff = cv2.absdiff(gray_prev, gray)

    # Apply thresholding to the difference to detect regions of motion
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Main loop to process the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame")
        break

    # If there's a previous frame, detect motion
    if prev_frame is not None:
        contours = detect_motion(frame, prev_frame)

        for contour in contours:
            # Ignore small movements
            if cv2.contourArea(contour) < 500:
                continue

            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Track the center point of the object
            cx = x + w // 2
            cy = y + h // 2
            trajectory_points.append((cx, cy))

    # Draw trajectory points on the frame
    for i in range(1, len(trajectory_points)):
        if trajectory_points[i - 1] is None or trajectory_points[i] is None:
            continue
        cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

    # Display the frame with detected objects and trajectory
    cv2.imshow("Object Trajectory Detection", frame)

    # Update the previous frame to the current one
    prev_frame = frame.copy()

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
cap.release()
cv2.destroyAllWindows()
