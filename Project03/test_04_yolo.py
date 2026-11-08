
import cv2

from ultralytics import YOLO
from config import YOLO_PATH, CONF_THRESHOLD, CLASS_MAP
from test_01_camera import get_camera


def detect_items():

    print("YOLO 모델 로딩 중...")

    model = YOLO(str(YOLO_PATH))
    cap = get_camera()

    if not cap.isOpened():
        raise RuntimeError("카메라 연결 실패")

    detected_items = set()

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("[FAIL] 프레임 읽기 실패")
                break

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            for result in results:
                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    if class_name in CLASS_MAP:
                        detected_items.add(
                            CLASS_MAP[class_name]
                        )

            annotated = results[0].plot()

            cv2.imshow("YOLO Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n탐지된 준비물:", sorted(detected_items))
    print("[OK] YOLO 실행 종료")

    return detected_items


if __name__ == "__main__":
    detect_items()
