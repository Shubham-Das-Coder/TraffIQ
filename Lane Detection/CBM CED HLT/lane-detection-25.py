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

    # Segment grayscale into groups
    num_groups = 8
    group_size = 256 // num_groups
    groups = []
    group_vis = np.zeros_like(gray)

    for i in range(num_groups):
        lower = i * group_size
        upper = 255 if i == num_groups - 1 else (i + 1) * group_size - 1
        mask = cv2.inRange(gray, lower, upper)
        groups.append(mask)
        group_vis[mask > 0] = (i + 1) * 30  # Visualization shade

    # Select last 2 groups (brightest areas for lane)
    lane_mask = cv2.bitwise_or(groups[-1], groups[-2])

    # Define ROI triangle
    y_bottom = int(0.85 * height)
    pts = np.array([
        [0, y_bottom],
        [width, y_bottom],
        [width // 2, height // 2]
    ], np.int32)

    mask_roi = np.zeros_like(lane_mask)
    cv2.fillPoly(mask_roi, [pts], 255)

    # Apply ROI to selected mask
    masked = cv2.bitwise_and(lane_mask, mask_roi)

    # Morphological filtering
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    rect_blocks = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_rect, iterations=1)
    solid_lines = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    final_mask = cv2.bitwise_or(rect_blocks, solid_lines)

    # Draw contours on copy of original frame
    output_frame = frame.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Draw ROI triangle
    cv2.polylines(output_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Convert all grayscale views to BGR for display stacking
    group_vis_color = cv2.cvtColor(group_vis, cv2.COLOR_GRAY2BGR)
    lane_mask_color = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    final_mask_color = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    # Resize everything to same size
    h, w = 360, 640
    output_frame = cv2.resize(output_frame, (w, h))
    group_vis_color = cv2.resize(group_vis_color, (w, h))
    lane_mask_color = cv2.resize(lane_mask_color, (w, h))
    final_mask_color = cv2.resize(final_mask_color, (w, h))

    # Stack 2x2 grid
    top_row = np.hstack((output_frame, group_vis_color))
    bottom_row = np.hstack((lane_mask_color, final_mask_color))
    all_views = np.vstack((top_row, bottom_row))

    # Show all combined
    cv2.imshow("Lane Detection Pipeline (Grouped View)", all_views)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
