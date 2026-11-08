
import pyttsx3


def text_to_speech(text):

    engine = pyttsx3.init()

    voices = engine.getProperty("voices")

    print("[설치된 음성 목록]")

    korean_found = False

    for voice in voices:

        print(voice.id, voice.name)

        description = (
            str(voice.id) + " " + str(voice.name)
        ).lower()

        if not korean_found and (
            "korean" in description or "ko_" in description
        ):
            engine.setProperty("voice", voice.id)
            korean_found = True

    if not korean_found:
        print("[주의] 한국어 음성을 찾지 못했습니다.")

    engine.setProperty("rate", 150)

    print("TTS 안내:", text)

    engine.say(text)
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":

    text_to_speech(
        "노트북과 휴대폰이 아직 확인되지 않았습니다."
    )

    print("[OK] TTS 실행 종료")
