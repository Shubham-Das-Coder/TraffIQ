# lane_detection_canny.py
import cv2
import numpy as np

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, height), (image.shape[1], height), (image.shape[1]//2, int(height*0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 5)
    return line_image

cap = cv2.VideoCapture("D:/Shubham/1101-143052492.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    line_image = display_lines(frame, lines)
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Lane Detection", combo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
