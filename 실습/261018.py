import cv2
import numpy as np
from numpy.linalg import inv
def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
while True:
    ret, frame = cap.read()
    if not ret:
            break
    gaussian_blur = cv2.GaussianBlur(frame, (5,5), 0) 
    ball_lab =cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2LAB)
    ball_mask = cv2.inRange(ball_lab, green_lower, green_upper)
    ball_mask=cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
    ball_mask=cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
    
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

    
    cv2.imshow("VideoCapture with Ball", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
     break

cap.release()
cv2.destroyAllWindows()


