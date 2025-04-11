import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Focus only on the bottom half in a trapezoid
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 35 / 100), int(height * 0.55)),
        (int(width * 65 / 100), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip vertically if needed
    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White and Yellow lane detection
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 80, 140])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Grayscale and blur
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Crop only lower trapezoid
    roi = region_of_interest(edges)

    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:  # Ignore small noisy lines
            # Fit a polynomial to avoid zigzags
            cnt = cnt.squeeze()
            if cnt.ndim == 2 and len(cnt) >= 5:
                # Sort by y-coordinate to get a smooth curve
                cnt = cnt[np.argsort(cnt[:, 1])]

                # Fit a poly line
                approx = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), 5, False)
                for i in range(len(approx) - 1):
                    pt1 = tuple(approx[i][0])
                    pt2 = tuple(approx[i + 1][0])
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 4)  # Blue lines

    cv2.imshow("Clean Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
