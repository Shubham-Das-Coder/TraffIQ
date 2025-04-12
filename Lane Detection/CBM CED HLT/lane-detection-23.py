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

    # Create grayscale classes (e.g., 5 classes)
    num_classes = 5
    class_range = 256 // num_classes
    classes = np.zeros_like(gray)

    for i in range(num_classes):
        lower = i * class_range
        upper = (i + 1) * class_range
        mask = cv2.inRange(gray, lower, upper - 1)
        classes[mask > 0] = (i + 1) * 50  # for visualization

    # Use the brightest 1-2 classes for lane detection
    lane_mask = cv2.inRange(gray, 180, 255)

    # Create a triangular ROI mask
    y_bottom = int(0.85 * height)
    pts = np.array([
        [0, y_bottom],
        [width, y_bottom],
        [width // 2, height // 2]
    ], np.int32)

    mask = np.zeros_like(lane_mask)
    cv2.fillPoly(mask, [pts], 255)

    # Apply ROI to lane mask
    masked = cv2.bitwise_and(lane_mask, mask)

    # Morphological operations
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

    # Draw ROI triangle
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Show outputs
    cv2.imshow("Lane Detection", frame)
    cv2.imshow("Grayscale Classes", classes)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
