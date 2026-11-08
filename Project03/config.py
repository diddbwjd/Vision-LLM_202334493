
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# TODO: Jetson 실제 모델 경로 확인
GEMMA_PATH = BASE_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
YOLO_PATH = BASE_DIR / "src/models/yolov8n.pt"

# TODO: CSI 카메라면 "csi", USB 카메라면 "usb"
CAMERA_TYPE = "usb"
CAMERA_INDEX = 0

# TODO: Whisper 모델이 설치되어 있는지 확인
STT_MODEL = "tiny"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
DETECTION_SECONDS = 20
CONF_THRESHOLD = 0.4

CLASS_MAP = {
    "laptop": "노트북",
    "cell phone": "휴대폰",
    "book": "책",
    "backpack": "가방",
    "handbag": "핸드백",
    "bottle": "물병",
    "umbrella": "우산",
    "mouse": "마우스",
    "keyboard": "키보드",
}
