#Project 01

import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
# 공 색깔
green_lower = np.array([30, 60, 90], dtype=np.uint8) 
green_upper = np.array([230, 115, 180], dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#공 위치 추적
prev_center = None

while True:
    ret, frame = cap.read()

    if not ret:
        break
    #공 색상 검출 및 전처리 과정
    gaussian_blur = cv2.GaussianBlur(frame, (5,5), 0) 
    ball_lab =cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2LAB)
    ball_mask = cv2.inRange(ball_lab, green_lower, green_upper)
    ball_mask=cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask=cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
    
    #공 윤곽 검출
    contours, _ = cv2.findContours(
    ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour)>500:
                (x,y),radius = cv2.minEnclosingCircle(largest_contour)
                center= (int(x),int (y))
                radius=int(radius)
                detected=True
                cv2.circle(frame,center,radius,(0,255,0),2)
                
                # 공의 움직임 추적
                if prev_center is not None:
                    y_change = center[1] - prev_center[1]
                    if y_change < -3:
                        direction = "UP"
                    elif y_change > 3:
                        direction = "DOWN"
                    else:
                        direction = "STILL"
            prev_center = center
                
                


    cv2.imshow("VideoCapture with GStreamer", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()