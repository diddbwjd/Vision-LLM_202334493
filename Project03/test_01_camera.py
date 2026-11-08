import cv2

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def get_camera():
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def main():
    cap = get_camera()

    if not cap.isOpened():
        print("[FAIL] 카메라 연결 실패")
        return

    print("[OK] 카메라 연결 성공")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[FAIL] 프레임 읽기 실패")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[OK] 카메라 테스트 완료")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()