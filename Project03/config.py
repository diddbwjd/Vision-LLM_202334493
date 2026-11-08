
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# TODO: Jetson 실제 모델 경로 확인
GEMMA_PATH = BASE_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
YOLO_PATH = BASE_DIR / "src/models/yololln_int8.engin"

# TODO: CSI 카메라면 "csi", USB 카메라면 "usb"
CAMERA_TYPE = "usb"
CAMERA_INDEX = 0

# Whisper.cpp 실행 파일
WHISPER_CLI_PATH = BASE_DIR / "whisper.cpp/build-cpu/bin/whisper-cli"

# Whisper 모델 파일
WHISPER_MODEL_PATH = BASE_DIR / "whisper.cpp/models/ggml-base.bin"

CLASS_MAP = {
    "laptop": "노트북",
    "cell phone": "휴대폰",
    "book": "책",
    "backpack": "가방",
    "bottle": "물병",
    "mouse": "마우스",
    "keyboard": "키보드",
}
