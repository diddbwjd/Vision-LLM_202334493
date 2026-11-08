
import json
import re

from llama_cpp import Llama
from config import GEMMA_PATH


def recommend_items(purpose):

    if not GEMMA_PATH.is_file():
        raise FileNotFoundError(GEMMA_PATH)

    print("Gemma 모델 로딩 중...")

    llm = Llama(
        model_path=str(GEMMA_PATH),
        n_gpu_layers=-1,  # Jetson GPU 가속 빌드 필요
        n_ctx=2048,
        n_batch=32,
        n_ubatch=32,
        verbose=True
    )

    prompt = f"""
외출 목적: {purpose}

외출에 필요한 준비물 3~5개를 추천해줘.
한국어 준비물 이름만 JSON 배열로 출력해.

예시:
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

    answer = response["choices"][0]["message"]["content"]

    print("\nGemma 원본 응답:", answer)

    match = re.search(r"\[[\s\S]*?\]", answer or "")

    if match is None:
        raise ValueError("JSON 배열을 찾을 수 없습니다.")

    items = json.loads(match.group())

    if not isinstance(items, list):
        raise ValueError("JSON 배열 형식이 아닙니다.")

    return items


if __name__ == "__main__":

    purpose = "오늘 학교에 가서 수업을 듣고 카페에서 과제를 할 거야."

    items = recommend_items(purpose)

    print("\n[추천 준비물]")
    for item in items:
        print("-", item)

    print("[OK] Gemma 실행 성공")
