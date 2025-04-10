# lane_detection_color_angle_filtered.py
import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("D:/Shubham/1101-143052492.mp4")  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for white and yellow lanes
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    filtered = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Edge detection
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle of the line in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            # Filter lines that are vertical to ±45 degrees from vertical
            if 45 <= abs(angle) <= 135:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Optional: Print the angle for debugging
                # print(f"Angle: {angle:.2f}")

    # Display the result
    cv2.imshow("Filtered Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
