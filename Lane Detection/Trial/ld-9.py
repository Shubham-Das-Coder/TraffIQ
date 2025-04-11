import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width*0.6), int(height*0.6)),
        (int(width*0.4), int(height*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def preprocess(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def enhance_mask(masked_frame):
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    blurred = cv2.GaussianBlur(closed, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_lines(edges, frame):
    cropped = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    line_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0), 5)
    return line_img

def fallback_contours(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cont_img, (x,y), (x+w, y+h), (255,0,0), 2)
    return cont_img

# ---------- Video Pipeline ----------
video_path = "D:/Shubham/1101-143052492.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video file: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended or failed to read frame.")
        break

    frame = cv2.resize(frame, (960, 540))

    masked = preprocess(frame)
    edges = enhance_mask(masked)
    lines_img = detect_lines(edges, frame)

    result = cv2.addWeighted(frame, 0.8, lines_img, 1, 1)

    # Fallback contour check
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([18, 94, 140]), np.array([48, 255, 255]))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    fallback_img = fallback_contours(result, combined_mask)

    cv2.imshow("Robust Lane Detection", fallback_img)

    # Exit if user presses 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
