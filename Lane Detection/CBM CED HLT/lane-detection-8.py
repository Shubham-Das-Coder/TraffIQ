import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Bottom trapezoid (ignore upper half)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.4), int(height * 0.55)),
        (int(width * 0.6), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def smooth_contour(cnt):
    cnt = cnt.squeeze()
    if cnt.ndim != 2 or len(cnt) < 5:
        return None

    # Sort points by y-coordinate
    cnt = cnt[np.argsort(cnt[:, 1])]

    # Fit a polynomial curve (2nd degree)
    y_vals = cnt[:, 1]
    x_vals = cnt[:, 0]
    z = np.polyfit(y_vals, x_vals, 2)
    p = np.poly1d(z)

    y_new = np.linspace(y_vals.min(), y_vals.max(), 100)
    x_new = p(y_new)

    # Stack into points
    points = np.array([np.stack((x_new, y_new), axis=1)], dtype=np.int32)
    return points

cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White mask
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Yellow mask
    lower_yellow = np.array([15, 80, 140])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    cropped_edges = region_of_interest(edges)

    contours, _ = cv2.findContours(cropped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small noisy ones
            smooth = smooth_contour(cnt)
            if smooth is not None:
                cv2.polylines(frame, [smooth], isClosed=False, color=(0, 0, 255), thickness=4)

    # Optional: Show cropped ROI for debugging
    # cv2.imshow("Edges ROI", cropped_edges)

    cv2.imshow("Final Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
