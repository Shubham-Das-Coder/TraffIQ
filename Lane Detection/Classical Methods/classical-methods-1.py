# Trying all the basic methods for lane detection

import cv2
import numpy as np
import pyautogui

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

# Create full-screen window
cv2.namedWindow("Lane Detection Techniques (3x3 Grid)", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lane Detection Techniques (3x3 Grid)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def region_of_interest(img):
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    output = []

    # 1. Grayscale Thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    output.append(cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR))

    # 2. Color Filtering (in HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)
    output.append(cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR))

    # 3. Edge Detection (Canny)
    edges = cv2.Canny(gray, 50, 150)
    output.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    # 4. Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    hough_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(hough_img, (x1,y1), (x2,y2), (0,255,0), 2)
    output.append(hough_img)

    # 5. Perspective Transform (Bird’s Eye View)
    height, width = gray.shape
    src = np.float32([[100, height], [540, height], [400, 250], [240, 250]])
    dst = np.float32([[100, height], [540, height], [540, 0], [100, 0]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    output.append(warped)

    # 6. Histogram Peaks
    binary = thresh_gray
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    hist_img = np.zeros_like(frame)
    for x, val in enumerate(histogram):
        cv2.line(hist_img, (x, frame.shape[0]), (x, frame.shape[0]-val//255), (255, 255, 255), 1)
    output.append(hist_img)

    # 7. Sliding Window Method
    out_img = np.dstack((binary, binary, binary))
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int32(binary.shape[0]//nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window+1)*window_height
        win_y_high = binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
    
    output.append(out_img)

    # 8. Contours + Shape Filtering
    contours_img = frame.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 1000:
            cv2.drawContours(contours_img, [cnt], -1, (0,255,0), 2)
    output.append(contours_img)

    # 9. Gradient-based Thresholding (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    _, sobel_thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    output.append(cv2.cvtColor(sobel_thresh, cv2.COLOR_GRAY2BGR))

    # Resize all images to same size for stacking
    resized = [cv2.resize(img, (426, 240)) for img in output]

    # Arrange in 3x3 grid
    top = np.hstack(resized[0:3])
    mid = np.hstack(resized[3:6])
    bottom = np.hstack(resized[6:9])
    grid_view = np.vstack([top, mid, bottom])

    # Resize to full screen and display
    full_screen_view = cv2.resize(grid_view, (screen_width, screen_height))
    cv2.imshow("Lane Detection Techniques (3x3 Grid)", full_screen_view)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
