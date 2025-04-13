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

    # Apply Gaussian blur to reduce noise from sunlight
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # --- Method 1: Grayscale Thresholding ---
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # --- Method 2: HSV Color Filtering ---
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # --- Combine both methods ---
    combined_lane = cv2.bitwise_and(thresh_gray, mask_hsv)

    # --- Morphological Opening to remove small noise ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_lane = cv2.morphologyEx(combined_lane, cv2.MORPH_OPEN, kernel)

    # --- Define trapezoidal ROI ---
    y_bottom = int(0.85 * height)
    y_top = int(y_bottom - 0.3 * height)
    x_left = int(0.4 * width)
    x_right = int(0.6 * width)

    pts = np.array([
        [0, y_bottom],             # Bottom left
        [width, y_bottom],         # Bottom right
        [x_right, y_top],          # Top right (40% from right)
        [x_left, y_top]            # Top left (40% from left)
    ], np.int32)

    mask_roi = np.zeros_like(combined_lane)
    cv2.fillPoly(mask_roi, [pts], 255)

    # --- Apply ROI ---
    masked_lane = cv2.bitwise_and(combined_lane, mask_roi)

    # --- Detect contours and filter by width ---
    contours, _ = cv2.findContours(masked_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_highlight = frame.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 6 <= w <= 50:
            cv2.drawContours(lane_highlight, [cnt], -1, (0, 0, 255), -1)

    # --- Draw ROI outline ---
    cv2.polylines(lane_highlight, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Prepare resized images for display
    original_resized = cv2.resize(frame, (320, 180))  # Upper left
    final_resized = cv2.resize(lane_highlight, (320, 180))  # Upper right
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
