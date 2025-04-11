
# Very Unique But Can Be Improved

import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    # Ignore top 50% of the frame
    roi_offset = height // 2
    roi = frame[roi_offset:, :]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Binary threshold to isolate white markings
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to enhance solid lines and blocks
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # One for detecting rectangular dashed blocks
    rect_blocks = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_rect, iterations=1)

    # Another for continuous solid lines
    solid_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line, iterations=2)

    # Combine both detections
    combined = cv2.bitwise_or(rect_blocks, solid_lines)

    # Find contours of the detected markings
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:  # filter small noise
            continue

        # Draw the detected marking (contour) in green on the original frame
        cv2.drawContours(frame[roi_offset:], [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Draw a red line indicating ignored region
    cv2.line(frame, (0, roi_offset), (width, roi_offset), (0, 0, 255), 2)

    # Display result
    cv2.imshow("Lane Markings Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
