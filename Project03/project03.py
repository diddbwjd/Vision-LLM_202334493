
import cv2
import json
import re
import gc
import sounddevice as sd
import pyttsx3

from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama
from ultralytics import YOLO


# ========================================
# 1. 공통 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent

GEMMA_PATH = BASE_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
YOLO_PATH = BASE_DIR / "src/models/yolov8n.pt"

STT_MODEL = "tiny"

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

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

# USB 카메라 파이프라인
# Jetson에서 정상 작동했던 파이프라인으로 교체
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# ========================================
# 2. Whisper 음성 인식
# ========================================

def speech_to_text():

    print("\n[1] 음성 인식 시작")
    input("Enter를 누르면 녹음합니다.")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

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

    print("인식 결과:", text)

    return text


# ========================================
# 3. Gemma 준비물 추천
# ========================================

def recommend_items(purpose):

    print("\n[2] 준비물 추천 시작")

    llm = Llama(
        model_path=str(GEMMA_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=32,
        n_ubatch=32,
        verbose=False
    )

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 외출 준비물 추천 도우미야. "
                    "사용자의 외출 목적에 맞는 준비물 3~5개를 추천해. "
                    "반드시 JSON 문자열 배열만 출력해. "
                    '예: ["노트북", "휴대폰", "책"]'
                )
            },
            {
                "role": "user",
                "content": purpose
            }
        ],
        max_tokens=150,
        temperature=0.3
    )

    answer = response["choices"][0]["message"]["content"]

    print("Gemma 응답:", answer)

    match = re.search(r"\[[\s\S]*?\]", answer or "")

    if match is None:
        raise ValueError("Gemma 응답에서 준비물 목록을 찾지 못했습니다.")

    items = json.loads(match.group())

    if not isinstance(items, list) or not all(
        isinstance(item, str) for item in items
    ):
        raise ValueError("준비물 목록 형식이 올바르지 않습니다.")

    items = list(dict.fromkeys(
        item.strip() for item in items if item.strip()
    ))

    # Gemma 사용 후 메모리 정리
    del llm
    gc.collect()

    print("추천 준비물:", items)

    return items


# ========================================
# 4. 준비물 목록 수정
# ========================================

def edit_items(items):

    print("\n[3] 준비물 목록 수정")

    while True:
        print("\n현재 준비물:", items)
        print("1. 준비물 추가")
        print("2. 준비물 삭제")
        print("3. 수정 완료")

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
            print("올바른 번호를 입력하세요.")

    return items


# ========================================
# 5. YOLO 객체 탐지
# ========================================

def detect_items():

    print("\n[4] YOLO 객체 탐지 시작")

    model = YOLO(str(YOLO_PATH))

    cap = cv2.VideoCapture(
        pipeline,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        raise RuntimeError("USB 카메라 연결 실패")

    detected_items = set()

    print("카메라 화면에서 준비물을 보여주세요.")
    print("탐지를 끝내려면 q를 누르세요.")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("프레임 읽기 실패")
                break

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            result = results[0]

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in CLASS_MAP:
                    detected_items.add(
                        CLASS_MAP[class_name]
                    )

            annotated_frame = result.plot()

            cv2.imshow(
                "ReadyMate - YOLO",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("탐지 결과:", detected_items)

    return detected_items


# ========================================
# 6. 준비물 비교
# ========================================

def compare_items(items, detected_items):

    print("\n[5] 준비물 비교")

    # 이름이 다른 동일 물건을 통일
    aliases = {
        "핸드폰": "휴대폰",
        "스마트폰": "휴대폰",
        "생수": "물병",
        "백팩": "가방",
    }

    def normalize(item):
        item = item.strip()
        return aliases.get(item, item)

    normalized_detected = {
        normalize(item) for item in detected_items
    }

    supported_items = set(CLASS_MAP.values())

    confirmed = []
    missing = []
    unsupported = []

    for item in items:

        normalized = normalize(item)

        if normalized not in supported_items:
            unsupported.append(item)

        elif normalized in normalized_detected:
            confirmed.append(item)

        else:
            missing.append(item)

    print("확인된 준비물:", confirmed)
    print("미확인 준비물:", missing)
    print("자동 탐지 불가:", unsupported)

    return confirmed, missing, unsupported


# ========================================
# 7. TTS 음성 안내
# ========================================

def text_to_speech(missing, unsupported):

    print("\n[6] TTS 음성 안내")

    messages = []

    if missing:
        messages.append(
            ", ".join(missing) +
            "이 아직 확인되지 않았습니다."
        )
    else:
        messages.append(
            "자동 탐지 가능한 준비물은 모두 확인되었습니다."
        )

    if unsupported:
        messages.append(
            ", ".join(unsupported) +
            "은 카메라로 자동 확인할 수 없으니 직접 확인해주세요."
        )

    message = " ".join(messages)

    print("안내 문장:", message)

    engine = pyttsx3.init()

    for voice in engine.getProperty("voices"):
        voice_name = (voice.name or "").lower()
        voice_id = (voice.id or "").lower()

        if "korean" in voice_name or "ko_" in voice_id:
            engine.setProperty("voice", voice.id)
            break

    engine.say(message)
    engine.runAndWait()


# ========================================
# 8. 최종 실행
# ========================================

def main():

    print("===== ReadyMate 실행 =====")

    # 1. 음성 인식
    purpose = speech_to_text()

    if not purpose:
        print("[FAIL] 음성이 인식되지 않았습니다.")
        return

    # 2. 준비물 추천
    items = recommend_items(purpose)

    # 3. 목록 수정
    items = edit_items(items)

    if not items:
        print("준비물 목록이 비어 있습니다.")
        return

    # 4. YOLO 객체 탐지
    detected_items = detect_items()

    # 5. 준비물 비교
    confirmed, missing, unsupported = compare_items(
        items,
        detected_items
    )

    # 6. 음성 안내
    text_to_speech(missing, unsupported)

    print("\n===== ReadyMate 실행 완료 =====")


if __name__ == "__main__":
    main()
