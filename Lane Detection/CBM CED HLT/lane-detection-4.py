import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    # Focus on bottom 60% of the frame
    mask = np.zeros_like(img)

    # Define the region as a trapezoid
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.5)),
        (int(width * 0.6), int(height * 0.5)),
        (int(width * 0.9), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)  # Flip vertically if needed
    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lane color ranges
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Focus on region of interest
    roi = region_of_interest(edges)

    # Find contours instead of lines
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw only large, curved-like contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:  # Filter out small noise
            # Fit curve using polynomial or just draw smoothed contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    cv2.imshow("Curved Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
