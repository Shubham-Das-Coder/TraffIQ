import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define trapezoid covering lower half and wide area (left to right)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.35), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.55)),
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

    # Masks for white and yellow lanes
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 80, 140])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Convert to gray and blur
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Crop only the region of interest (lower half + trapezoid)
    cropped_edges = region_of_interest(edges)

    # Find contours
    contours, _ = cv2.findContours(cropped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            # Approximate to smooth the contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("Enhanced Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
