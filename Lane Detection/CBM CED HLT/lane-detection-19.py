# Modified ROI

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

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binary threshold to isolate white markings
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Create a triangular mask
    mask = np.zeros_like(binary)
    pts = np.array([[0, height], [width, height], [width // 2, height // 2]], np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Apply mask to binary image
    masked = cv2.bitwise_and(binary, mask)

    # Morphological operations to enhance solid lines and blocks
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    rect_blocks = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_rect, iterations=1)
    solid_lines = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_line, iterations=2)

    combined = cv2.bitwise_or(rect_blocks, solid_lines)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Optional: Draw ROI triangle on screen
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Show output
    cv2.imshow("Lane Markings Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
