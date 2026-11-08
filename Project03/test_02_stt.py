import sounddevice as sd
import subprocess
import wave
import tempfile

from pathlib import Path
from config import (
    SAMPLE_RATE,
    RECORD_SECONDS,
    
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
)


# Whisper.cpp를 이용한 음성 인식
def speech_to_text():

    print("[1] 마이크 장치 확인")
    print(sd.query_devices())

    # Whisper 실행 파일 확인
    if not WHISPER_CLI_PATH.is_file():
        print("[FAIL] whisper-cli 파일이 없습니다.")
        return ""

    # Whisper 모델 확인
    if not WHISPER_MODEL_PATH.is_file():
        print("[FAIL] Whisper 모델 파일이 없습니다.")
        return ""

    input("\nEnter를 누르면 녹음을 시작합니다.")

    print("[2] 녹음 중...")

    # 5초 동안 마이크 녹음
    audio = sd.rec(
        int(SAMPLE_RATE * RECORD_SECONDS),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    print("[OK] 녹음 완료")

    # 임시 WAV 파일 생성 및 Whisper 실행
    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = Path(temp_dir) / "record.wav"
        output_path = Path(temp_dir) / "result"

        # 녹음 데이터를 WAV 파일로 저장
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio.tobytes())

        print("[3] Whisper 음성 인식 시작")

        # Whisper.cpp 실행
        result = subprocess.run(
            [
                str(WHISPER_CLI_PATH),
                "-m", str(WHISPER_MODEL_PATH),
                "-f", str(wav_path),
                "-l", "ko",
                "-nt",
                "-otxt",
                "-of", str(output_path)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[FAIL] Whisper 실행 실패")
            print(result.stderr)
            return ""

        # Whisper가 생성한 텍스트 파일 읽기
        txt_path = Path(str(output_path) + ".txt")

        if not txt_path.is_file():
            print("[FAIL] 인식 결과 파일이 없습니다.")
            return ""

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

    print("[4] 음성 인식 결과")
    print(text)

    if text:
        print("[OK] STT 테스트 성공")
    else:
        print("[FAIL] 음성 인식 결과 없음")

    return text


if __name__ == "__main__":
    speech_to_text()
