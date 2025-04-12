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

    # Divide grayscale into 8 groups (0-32, 32-64, ..., 224-255)
    num_groups = 8
    group_size = 256 // num_groups
    groups = []

    # For visualization
    group_vis = np.zeros_like(gray)

    for i in range(num_groups):
        lower = i * group_size
        upper = 255 if i == num_groups - 1 else (i + 1) * group_size - 1
        mask = cv2.inRange(gray, lower, upper)
        groups.append(mask)
        group_vis[mask > 0] = (i + 1) * 30  # Visualize each group as a shade

    # Select only last two groups (brightest bands)
    lane_mask = cv2.bitwise_or(groups[-1], groups[-2])  # groups[6] + groups[7]

    # Define triangular ROI
    y_bottom = int(0.85 * height)
    pts = np.array([
        [0, y_bottom],
        [width, y_bottom],
        [width // 2, height // 2]
    ], np.int32)

    mask_roi = np.zeros_like(lane_mask)
    cv2.fillPoly(mask_roi, [pts], 255)

    # Apply ROI
    masked = cv2.bitwise_and(lane_mask, mask_roi)

    # Morphological cleaning
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

    # Draw ROI outline
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Show outputs
    cv2.imshow("Lane Detection", frame)
    cv2.imshow("Grayscale Groups", group_vis)
    cv2.imshow("Selected Lane Mask", lane_mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
