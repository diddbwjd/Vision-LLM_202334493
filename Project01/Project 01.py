#Project 01
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
# 공 색깔 범위 설정
green_lower = np.array([30, 60, 90], dtype=np.uint8) 
green_upper = np.array([230, 115, 180], dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#공 위치 추적(중심점 초기화)
prev_center = None

# 손 검출 (모델 가져오기 및 option 설정)
base_option = python.BaseOptions(
    model_asset_path="../실습/src/models/MediaPipe/hand_landmarker.task")
options = vision.HandLandmarkerOptions( base_options = base_option, num_hands=1 )
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

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
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour)>500:
                (x,y),radius = cv2.minEnclosingCircle(largest_contour)
                center= (int(x),int (y))
                radius=int(radius)
                detected=True
                #공 추적하여 그리기
                cv2.circle(frame,center,radius,(0,255,0),2)
                
                # 공의 움직임 추적(상하 움직임)
                if prev_center is not None:
                    y_change = center[1] - prev_center[1]
                    if y_change < -3:
                        direction = "UP"
                    elif y_change > 3:
                        direction = "DOWN"
                    else:
                        direction = "STILL"
            prev_center = center
    if not detected:
        prev_center = None
        
    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 손 검출
    result = hand_detector.detect(
    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) )
    
   # 손이 검출되었을 경우
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # 프레임 크기 가져오기
            h, w = frame.shape[:2]

        # 손의 21개 Landmark를 픽셀 좌표로 변환
        points = [
            (int(p.x * w), int(p.y * h))
            for p in hand
        ]

        # 손 관절 연결선 그리기
        for c in connections:
            cv2.line(
                frame,
                points[c.start],
                points[c.end],
                (0, 255, 0),
                2
            )

        # 손 관절 위치에 점 그리기
        for point in points:
            cv2.circle(frame, point, 4, (255, 0, 0), -1)
        hand_points = np.array(points, dtype=np.int32)
        hand_hull = cv2.convexHull(hand_points)
        cv2.polylines(frame, [hand_hull], True, (0, 255, 255), 2)
                


    cv2.imshow("VideoCapture with GStreamer", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()