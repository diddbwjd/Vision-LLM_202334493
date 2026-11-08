
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 별도 가상환경에 설치한 Piper 실행 파일
PIPER_PATH = BASE_DIR / ".piper_venv/bin/piper"

# 한국어 여자 목소리 모델
PIPER_MODEL = BASE_DIR / "src/models/Piper/ko_KR-kss-medium.onnx"

# 생성할 음성 파일
OUTPUT_PATH = Path("/tmp/piper_test.wav")


def text_to_speech(text):
    print("TTS 안내:", text)

    # Piper 실행 파일 확인
    if not PIPER_PATH.is_file():
        raise FileNotFoundError(f"Piper 실행 파일 없음: {PIPER_PATH}")

    # 한국어 모델 확인
    if not PIPER_MODEL.is_file():
        raise FileNotFoundError(f"한국어 모델 없음: {PIPER_MODEL}")

    # 한국어 음성 생성
    subprocess.run(
        [
            str(PIPER_PATH),
            "--model", str(PIPER_MODEL),
            "--output_file", str(OUTPUT_PATH)
        ],
        input=text,
        text=True,
        check=True
    )

    # 음성 재생
    subprocess.run(
        ["aplay", str(OUTPUT_PATH)],
        check=True
    )


if __name__ == "__main__":
    text_to_speech("노트북과 휴대폰이 아직 확인되지 않았습니다.")
    print("[OK] 한국어 TTS 실행 성공")
