# improved_lane_detection.py
import cv2
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0, height), (width, height), (width//2 + 50, int(height*0.55)), (width//2 - 50, int(height*0.55))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 8)
    return line_image

def preprocess_frame(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return masked_frame

cap = cv2.VideoCapture("D:/Shubham/1101-143052492.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional for faster processing)
    frame = cv2.resize(frame, (960, 540))

    # Preprocessing
    filtered = preprocess_frame(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI + Hough
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=120)
    line_image = display_lines(frame, lines)
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Improved Lane Detection", final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
