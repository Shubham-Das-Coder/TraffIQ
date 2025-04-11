import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define a trapezoid that only includes lower half of frame
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.4), int(height * 0.55)),
        (int(width * 0.6), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def smooth_and_validate_contour(cnt):
    cnt = cnt.squeeze()
    if cnt.ndim != 2 or len(cnt) < 5:
        return None

    # Sort by Y (top to bottom)
    cnt = cnt[np.argsort(cnt[:, 1])]

    y_vals = cnt[:, 1]
    x_vals = cnt[:, 0]

    # Check if x-values go consistently left or right
    dx = np.diff(x_vals)
    direction = np.sign(dx)
    if np.abs(np.sum(direction)) < len(dx) * 0.7:
        return None  # too much oscillation, not a lane

    # Fit a curve (2nd degree)
    z = np.polyfit(y_vals, x_vals, 2)
    p = np.poly1d(z)

    y_new = np.linspace(y_vals.min(), y_vals.max(), 100)
    x_new = p(y_new)

    # Discard curves that bend too sharply
    curvature = np.max(np.abs(np.diff(x_new, 2)))
    if curvature > 1.0:  # curvature threshold
        return None

    points = np.array([np.stack((x_new, y_new), axis=1)], dtype=np.int32)
    return points

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tighter color masks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked = cv2.bitwise_and(frame, frame, mask=combined_mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    roi_edges = region_of_interest(edges)

    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            smooth = smooth_and_validate_contour(cnt)
            if smooth is not None:
                cv2.polylines(frame, [smooth], isClosed=False, color=(0, 0, 255), thickness=4)

    # Draw blue boundary for region of interest
    overlay = frame.copy()
    roi_mask = np.zeros_like(frame)
    roi_polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.4), int(height * 0.55)),
        (int(width * 0.6), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], np.int32)
    cv2.polylines(roi_mask, roi_polygon, isClosed=True, color=(255, 0, 0), thickness=2)
    frame = cv2.addWeighted(overlay, 1, roi_mask, 1, 0)

    cv2.imshow("Lane Detection - Cleaned", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
