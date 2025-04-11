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
    roi_offset = int(height * 0.4)
    roi = frame[roi_offset:, :]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get white lanes on dark road
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # Morphology to connect broken lines and highlight thick areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of potential lane blocks
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare mask to draw detected blocks
    mask = np.zeros_like(roi)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # ignore small blobs
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Apply perspective-aware filtering
        abs_y = y + roi_offset
        thickness_ratio = h / w if w != 0 else 0

        # Allow thicker regions at bottom, thinner at top
        if abs_y > roi_offset + (height - roi_offset) * 0.5:
            min_thickness_ratio = 0.2  # near camera (thicker)
        else:
            min_thickness_ratio = 0.05  # far (thinner)

        if 0.3 < w / h < 10 and thickness_ratio > min_thickness_ratio:
            # Draw filled red box on the original frame
            cv2.rectangle(frame, (x, y + roi_offset), (x + w, y + h + roi_offset), (0, 0, 255), 2)
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

    # Draw red line to separate ignored area
    cv2.line(frame, (0, roi_offset), (width, roi_offset), (0, 0, 255), 2)

    # Show result
    cv2.imshow("Lane Block Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
