# Methods Used: 1, 2

import cv2
import numpy as np
import pyautogui

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

# Create full-screen window
cv2.namedWindow("Lane Detection - Combined Methods", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lane Detection - Combined Methods", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip vertically and resize
    frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame, (640, 360))

    # --- Method 1: Grayscale Thresholding ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # --- Method 2: Color Filtering in HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # --- Combine Method 1 & 2: Bitwise AND ---
    combined_lane = cv2.bitwise_and(thresh_gray, mask_hsv)

    # --- Mark the common region as lane (Blue on Original Frame) ---
    lane_mask = cv2.cvtColor(combined_lane, cv2.COLOR_GRAY2BGR)
    blue_overlay = np.zeros_like(frame)
    blue_overlay[:] = (255, 0, 0)  # Blue in BGR

    # Apply blue only where combined_lane is white
    lane_highlight = np.where(lane_mask == 255, blue_overlay, frame)

    # Resize to full screen and display
    full_screen_view = cv2.resize(lane_highlight.astype(np.uint8), (screen_width, screen_height))
    cv2.imshow("Lane Detection - Combined Methods", full_screen_view)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
