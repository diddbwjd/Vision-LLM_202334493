
import sounddevice as sd
from faster_whisper import WhisperModel

from config import SAMPLE_RATE, RECORD_SECONDS, STT_MODEL


def speech_to_text():

    print("마이크 장치 목록:")
    print(sd.query_devices())

    input("\nEnter를 누르면 녹음 시작...")

    print("녹음 중...")

    audio = sd.rec(
        int(SAMPLE_RATE * RECORD_SECONDS),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("[OK] 녹음 완료")

    # Jetson CPU에서 STT 실행
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

    print("STT 결과:", text)

    if text:
        print("[OK] STT 실행 성공")
    else:
        print("[FAIL] 음성 인식 결과 없음")

    return text


if __name__ == "__main__":
    speech_to_text()
