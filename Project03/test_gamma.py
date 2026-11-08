from pathlib import Path
from llama_cpp import Llama

# 다운로드한 Gemma 모델 경로
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)

print("모델 경로:", MODEL_PATH)
print("모델 크기:", round(MODEL_PATH.stat().st_size / 1024**3, 2), "GB")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=0,
    n_ctx=512,
    n_batch=32,
    n_ubatch=32,
    verbose=True
)

# Gemma에 질문하기
response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "학교에 갈 때 필요한 준비물 5개를 한국어로 알려줘."
        }
    ],
    max_tokens=200
)

# 응답 출력
print(response["choices"][0]["message"]["content"])