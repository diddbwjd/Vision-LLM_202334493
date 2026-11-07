
import gc
import json
import re
import time
from pathlib import Path

import cv2
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel
from llama_cpp import Llama
from ultralytics import YOLO


# ==========================================
# 1. 환경 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# TODO: Jetson에 저장된 실제 모델 경로 확인
GEMMA_PATH = BASE_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

# TODO: Jetson에 저장된 YOLO 모델 경로 확인
# TensorRT 엔진이 있으면 .engine 경로로 변경 가능
YOLO_PATH = BASE_DIR / "src/models/yolov8n.pt"

# TODO: Whisper 모델이 로컬에 있으면 해당 폴더 경로로 변경
# "tiny"는 첫 실행 시 모델을 다운로드할 수 있음
STT_MODEL = "tiny"

# 카메라 설정: "csi" 또는 "usb"
CAMERA_TYPE = "csi"

# TODO: USB 카메라를 사용하면 장치 번호 확인
CAMERA_INDEX = 0

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
DETECTION_SECONDS = 20
CONF_THRESHOLD = 0.4

# YOLO가 인식한 영어 이름을 한국어로 변환
# 일반 COCO YOLO 모델에서 지원하는 일부 클래스
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

# YOLO가 인식할 수 있는 준비물만 자동 확인
SUPPORTED_ITEMS = set(CLASS_MAP.values())


# ==========================================
# 2. Jetson 카메라 파이프라인
# ==========================================

def get_camera():

    if CAMERA_TYPE == "csi":

        # TODO: Jetson CSI 카메라용 파이프라인
        # 센서 종류와 지원 해상도에 따라 수정 필요
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), "
            "width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )

    else:
        # TODO: Jetson USB 카메라
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    return cap


# ==========================================
# 3. STT: 음성 입력
# ==========================================

def speech_to_text():

    print("\n외출 목적을 말해주세요.")
    print("5초 동안 녹음합니다.")

    # TODO: Jetson 마이크 장치 설정이 필요하면
    # sd.default.device = (입력장치번호, None)

    audio = sd.rec(
        int(SAMPLE_RATE * RECORD_SECONDS),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    print("녹음 완료!")

    # Whisper는 여기서 CPU로 실행
    # Jetson GPU는 Gemma와 YOLO에 사용
    model = WhisperModel(
        STT_MODEL,
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        audio.flatten(),
        language="ko",
        beam_size=1
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    del model
    gc.collect()

    print("인식된 문장:", text)

    return text


# ==========================================
# 4. Gemma: 준비물 추천
# ==========================================

def recommend_items(purpose):

    if not GEMMA_PATH.exists():
        raise FileNotFoundError(
            f"Gemma 모델 파일이 없습니다: {GEMMA_PATH}"
        )

    print("\nGemma 모델 실행 중...")

    llm = Llama(
        model_path=str(GEMMA_PATH),
        n_gpu_layers=-1,  # Jetson GPU 가속
        n_ctx=2048,
        n_batch=32,
        n_ubatch=32,
        verbose=False
    )

    prompt = f"""
너는 외출 준비를 도와주는 AI 비서야.

사용자의 외출 목적:
{purpose}

외출에 필요한 준비물 3~5개를 추천해줘.

규칙:
1. 한국어로 작성해.
2. 준비물 이름만 작성해.
3. 반드시 JSON 배열로 출력해.
4. 설명이나 다른 문장은 작성하지 마.

출력 예시:
["노트북", "책", "휴대폰"]
"""

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200,
        temperature=0.2
    )

    answer = (
        response["choices"][0]["message"]["content"]
        or ""
    ).strip()

    print("Gemma 응답:", answer)

    # JSON 배열 부분 추출
    match = re.search(r"\[[\s\S]*?\]", answer)

    if match is None:
        raise ValueError("Gemma가 준비물 배열을 생성하지 못했습니다.")

    items = json.loads(match.group())

    if not isinstance(items, list):
        raise ValueError("준비물 목록 형식이 올바르지 않습니다.")

    items = [
        item.strip()
        for item in items
        if isinstance(item, str) and item.strip()
    ]

    # 중복 제거
    items = list(dict.fromkeys(items))

    # 다음 단계에서 YOLO 메모리를 확보하기 위해 해제
    del llm
    gc.collect()

    return items


# ==========================================
# 5. 준비물 목록 출력 및 수정
# ==========================================

def show_items(items):

    print("\n[준비물 목록]")

    for i, item in enumerate(items, start=1):
        print(f"{i}. {item}")


def edit_items(items):

    while True:

        show_items(items)

        print("\n1. 준비물 추가")
        print("2. 준비물 삭제")
        print("3. 목록 확정")

        choice = input("선택: ").strip()

        if choice == "1":

            item = input("추가할 준비물: ").strip()

            if item and item not in items:
                items.append(item)

        elif choice == "2":

            item = input("삭제할 준비물: ").strip()

            if item in items:
                items.remove(item)
            else:
                print("목록에 없는 준비물입니다.")

        elif choice == "3":
            break

        else:
            print("다시 선택해주세요.")

    return items


# ==========================================
# 6. YOLO: 준비물 탐지
# ==========================================

def detect_items():

    print("\nYOLO 객체 탐지를 시작합니다.")

    model = YOLO(str(YOLO_PATH))

    cap = get_camera()

    detected_items = set()
    start_time = time.monotonic()

    try:
        while time.monotonic() - start_time < DETECTION_SECONDS:

            ret, frame = cap.read()

            if not ret:
                print("카메라 프레임을 읽지 못했습니다.")
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

                    # 영어 클래스 이름을 한국어로 변환
                    if class_name in CLASS_MAP:
                        korean_name = CLASS_MAP[class_name]
                        detected_items.add(korean_name)

            # YOLO 탐지 결과 화면
            annotated_frame = results[0].plot()

            cv2.imshow(
                "Outing Preparation - YOLO",
                annotated_frame
            )

            # q 키를 누르면 탐지 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        del model
        gc.collect()

    print("탐지된 준비물:", sorted(detected_items))

    return detected_items


# ==========================================
# 7. 준비물과 YOLO 탐지 결과 비교
# ==========================================

def compare_items(items, detected_items):

    found = []
    unconfirmed = []
    unsupported = []

    for item in items:

        if item not in SUPPORTED_ITEMS:
            unsupported.append(item)

        elif item in detected_items:
            found.append(item)

        else:
            unconfirmed.append(item)

    return found, unconfirmed, unsupported


# ==========================================
# 8. TTS: 음성 안내
# ==========================================

def text_to_speech(text):

    print("\n음성 안내:", text)

    engine = pyttsx3.init()

    # TODO: Jetson에 설치된 한국어 음성 확인
    # 한국어 음성이 없으면 정상 발음되지 않을 수 있음
    voices = engine.getProperty("voices")

    for voice in voices:
        description = (
            str(voice.id) + " " + str(voice.name)
        ).lower()

        if "korean" in description or "ko_" in description:
            engine.setProperty("voice", voice.id)
            break

    engine.setProperty("rate", 150)

    engine.say(text)
    engine.runAndWait()
    engine.stop()


# ==========================================
# 9. 전체 프로그램
# ==========================================

def main():

    print("================================")
    print("    외출 준비 도우미 시스템")
    print("================================")

    # 1. STT 음성 입력
    purpose = speech_to_text()

    # 음성 인식에 실패하면 키보드 입력으로 대체
    if not purpose:
        purpose = input("외출 목적을 직접 입력하세요: ").strip()

    if not purpose:
        print("외출 목적이 입력되지 않았습니다.")
        return

    # 2. Gemma 준비물 추천
    items = recommend_items(purpose)

    # 3. 사용자 준비물 수정
    items = edit_items(items)

    if not items:
        print("준비물 목록이 비어 있습니다.")
        return

    # 4. YOLO 카메라 탐지
    detected_items = detect_items()

    # 5. 준비물 비교
    found, unconfirmed, unsupported = compare_items(
        items,
        detected_items
    )

    print("\n[확인된 준비물]")
    print(found)

    print("\n[탐지되지 않은 준비물]")
    print(unconfirmed)

    print("\n[자동 확인을 지원하지 않는 준비물]")
    print(unsupported)

    # 6. TTS 음성 안내
    if unconfirmed:

        message = (
            ", ".join(unconfirmed)
            + "은 아직 확인되지 않았습니다."
        )

    else:
        message = "자동 확인 가능한 준비물 검사가 끝났습니다."

    if unsupported:
        message += (
            " 다음 물건은 직접 확인해주세요. "
            + ", ".join(unsupported)
        )

    text_to_speech(message)

    print("\n외출 준비 검사가 완료되었습니다.")


# ==========================================
# 프로그램 시작
# ==========================================

if __name__ == "__main__":
    main()
