import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

# Set the ROI ratio (ignore top 50%)
roi_ratio = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Fix orientation (rotate 180 degrees if video appears upside down)
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    height, width = frame.shape[:2]
    roi_offset = int(height * roi_ratio)

    # Draw a red line to indicate ignored top half
    cv2.line(frame, (0, roi_offset), (width, roi_offset), (0, 0, 255), 2)

    # Extract Region of Interest (only bottom half)
    roi = frame[roi_offset:, :]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to deal with varying light (sunlight on road)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-10
    )

    # Morphological operations to clean up thresholded image
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_rect)

    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_line, iterations=2)

    # Find contours in the processed mask
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 3000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        # Filter: long, narrow shapes (lane-like)
        if 0.2 < aspect_ratio < 3.5:
            cv2.drawContours(frame[roi_offset:], [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Show the result
    cv2.imshow("Lane Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
