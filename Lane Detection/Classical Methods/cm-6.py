# Combined methods 1 and 2, and using the triangular ROI

import cv2
import numpy as np
import pyautogui

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

# Create full-screen window
cv2.namedWindow("Lane Detection - ROI Combined", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lane Detection - ROI Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip vertically and resize
    frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame, (640, 360))
    height, width = frame.shape[:2]

    # --- Method 1: Grayscale Thresholding ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # --- Method 2: HSV Color Filtering ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # --- Combine both methods ---
    combined_lane = cv2.bitwise_and(thresh_gray, mask_hsv)

    # --- Define triangular ROI ---
    y_bottom = int(0.85 * height)
    pts = np.array([
        [0, y_bottom],
        [width, y_bottom],
        [width // 2, height // 2]
    ], np.int32)

    mask_roi = np.zeros_like(combined_lane)
    cv2.fillPoly(mask_roi, [pts], 255)

    # --- Apply ROI ---
    masked_lane = cv2.bitwise_and(combined_lane, mask_roi)

    # --- Highlight lanes in red on the original frame ---
    lane_highlight = frame.copy()
    lane_highlight[np.where(masked_lane == 255)] = (0, 0, 255)

    # --- Draw ROI outline ---
    cv2.polylines(lane_highlight, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Prepare resized images for display
    original_resized = cv2.resize(frame, (320, 180))  # Upper left: Original
    final_resized = cv2.resize(lane_highlight, (320, 180))  # Upper right: Final output with ROI
    method1_resized = cv2.resize(cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR), (320, 180))  # Lower left
    method2_resized = cv2.resize(cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR), (320, 180))  # Lower right

    # Stack images into a 2x2 grid
    top_row = np.hstack((original_resized, final_resized))
    bottom_row = np.hstack((method1_resized, method2_resized))
    grid_view = np.vstack((top_row, bottom_row))

    # Resize to full screen and display
    full_screen_view = cv2.resize(grid_view, (screen_width, screen_height))
    cv2.imshow("Lane Detection - ROI Combined", full_screen_view)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
