from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 친구 코드의 카메라 설정 유지: /dev/video0, JPEG 1280x720, 30fps
PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 기존 모델 경로. 파일 옆에 hand_landmarker.task를 두어도 실행 가능.
MODEL_PATH = "../실습/src/models/MediaPipe/hand_landmarker.task"

GREEN_LOWER = np.array([30, 60, 90], dtype=np.uint8)
GREEN_UPPER = np.array([230, 115, 180], dtype=np.uint8)
MIN_BALL_AREA = 500
MOVEMENT_THRESHOLD = 3


def find_model():
    """기존 실행 폴더 경로와 스크립트 기준 경로에서 모델 찾기."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(MODEL_PATH),
        script_dir / MODEL_PATH,
        script_dir / "hand_landmarker.task",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "손 모델을 찾을 수 없습니다. MODEL_PATH를 수정하거나 "
        "hand_landmarker.task를 이 파이썬 파일과 같은 폴더에 두세요."
    )


def main():
    model_path = find_model()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=1
    )
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    # 프레임 사이에 유지해야 하는 상태
    prev_center = None
    prev_touching = False
    prev_direction = "STILL"
    rebounded = False
    count = 0

    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    hand_detector = None

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "카메라 열기 실패: /dev/video0, JPEG 1280x720 30fps 지원 여부와 "
                "OpenCV의 GStreamer 지원을 확인하세요."
            )

        hand_detector = vision.HandLandmarker.create_from_options(options)
        print("실행 중: 영상 창에서 q 또는 ESC를 누르면 종료합니다.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽지 못했습니다. 카메라 형식과 연결을 확인하세요.")
                break

            # 반드시 매 프레임 초기화: 이전 검출 결과가 남지 않도록 함
            detected = False
            direction = "STILL"
            touching = False
            center = None
            radius = 0.0

            # 손 검출용 영상은 원/선/글자를 그리기 전에 별도로 준비
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 공 색상 검출: Gaussian 필터 -> LAB -> 마스크 -> 형태학 처리
            gaussian_blur = cv2.GaussianBlur(frame, (5, 5), 0)
            ball_lab = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2LAB)
            ball_mask = cv2.inRange(ball_lab, GREEN_LOWER, GREEN_UPPER)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_BALL_AREA:
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    center = (int(x), int(y))
                    detected = True

                    # 화면의 y 좌표는 아래로 갈수록 증가
                    if prev_center is not None:
                        y_change = center[1] - prev_center[1]
                        if y_change < -MOVEMENT_THRESHOLD:
                            direction = "UP"
                        elif y_change > MOVEMENT_THRESHOLD:
                            direction = "DOWN"

                        # 내려가다가 올라가기 시작하면 반동 후보로 기록
                        if prev_direction == "DOWN" and direction == "UP":
                            rebounded = True

                        if direction != "STILL":
                            prev_direction = direction

                    prev_center = center
                    cv2.circle(frame, center, max(1, int(radius)), (0, 255, 0), 2)

            if not detected:
                # 공을 놓치면 진행 중인 반동 판정을 취소.
                # 누적 count는 유지하고 새 검출부터 이동을 다시 계산.
                prev_center = None
                prev_direction = "STILL"
                rebounded = False

            result = hand_detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb)
                )
            )

            h, w = frame.shape[:2]
            for hand in result.hand_landmarks:
                points = [(int(p.x * w), int(p.y * h)) for p in hand]

                for connection in connections:
                    cv2.line(
                        frame, points[connection.start], points[connection.end],
                        (0, 255, 0), 2
                    )
                for point in points:
                    cv2.circle(frame, point, 4, (255, 0, 0), -1)

                hand_points = np.array(points, dtype=np.int32)
                hand_hull = cv2.convexHull(hand_points)
                cv2.polylines(frame, [hand_hull], True, (0, 255, 255), 2)

                # 중심이 손 영역 안이거나, 손 영역까지 거리가 공 반지름 이내
                if detected:
                    distance = cv2.pointPolygonTest(hand_hull, center, True)
                    if distance >= -radius:
                        touching = True

            # 손이 검출되지 않은 프레임에서도 관계 상태를 갱신
            new_touch = touching and not prev_touching
            if new_touch and rebounded:
                count += 1
                rebounded = False
            elif rebounded and direction == "DOWN" and not touching:
                # 기존 규칙: 반동 후 손에 닿지 않고 다시 내려오면 횟수 초기화
                count = 0
                rebounded = False
            prev_touching = touching

            cv2.putText(
                frame, f"Count: {count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Direction: {direction if detected else 'NOT FOUND'}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Touching: {touching}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cv2.imshow("Green Ball Bounce State", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        try:
            if hand_detector is not None:
                hand_detector.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
