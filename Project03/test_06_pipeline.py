
from test_02_stt import speech_to_text
from test_03_gemma import recommend_items
from test_04_yolo import detect_items
from test_05_tts import text_to_speech

from config import CLASS_MAP


def main():

    print("외출 준비 도우미 시스템 시작")

    # 1. 음성으로 외출 목적 입력
    purpose = speech_to_text()

    if not purpose:
        purpose = input("외출 목적 입력: ").strip()

    if not purpose:
        print("외출 목적이 없습니다.")
        return

    # 2. Gemma 준비물 추천
    items = recommend_items(purpose)

    # 3. 준비물 목록 확인 및 수정
    while True:

        print("\n[준비물 목록]")

        for i, item in enumerate(items, start=1):
            print(f"{i}. {item}")

        print("\n1. 추가  2. 삭제  3. 확정")

        choice = input("선택: ").strip()

        if choice == "1":
            item = input("추가할 물건: ").strip()

            if item and item not in items:
                items.append(item)

        elif choice == "2":
            item = input("삭제할 물건: ").strip()

            if item in items:
                items.remove(item)

        elif choice == "3":
            break

    if not items:
        print("준비물 목록이 비어 있습니다.")
        return

    # 4. YOLO 탐지
    detected = detect_items()

    # 5. 준비물 비교
    supported = set(CLASS_MAP.values())

    found = [
        item for item in items
        if item in supported and item in detected
    ]

    missing = [
        item for item in items
        if item in supported and item not in detected
    ]

    unsupported = [
        item for item in items
        if item not in supported
    ]

    print("\n확인된 준비물:", found)
    print("탐지되지 않은 준비물:", missing)
    print("직접 확인할 준비물:", unsupported)

    # 6. TTS 안내
    messages = []

    if missing:
        messages.append(
            ", ".join(missing) + "이 아직 확인되지 않았습니다."
        )

    if unsupported:
        messages.append(
            ", ".join(unsupported) + "은 직접 확인해주세요."
        )

    if not messages:
        messages.append("준비물 확인이 완료되었습니다.")

    text_to_speech(" ".join(messages))

    print("\n[OK] 전체 테스트 완료")


if __name__ == "__main__":
    main()
