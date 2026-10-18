import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 원본 이미지 불러오기
from pathlib import Path

# 현재 Python 파일 위치
BASE_DIR = Path(__file__).resolve().parent

# 실습 폴더의 이미지 경로
image_path = BASE_DIR.parent/ "실습" / "src" / "images" / "basketball_crop.jpg"

basketball = cv2.imread(str(image_path))

basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)
# LAB 색공간 변환 및 Binary Mask 생성
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

lower_lab = np.array([70, 130, 70])
upper_lab = np.array([180, 230, 185])

mask_lab = cv2.inRange(basketball_lab, lower_lab, upper_lab)


# TODO 1: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask_final = cv2.morphologyEx(mask_lab, cv2.MORPH_OPEN, kernel)


# TODO 2: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)


# TODO 3: 최종 농구공 영역 및 외곽선 선택
ball_mask = np.zeros_like(mask_final)
contour_result = basketball_rgb.copy()

if contours:
    ball_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(ball_mask, [ball_contour], -1, 255, -1)
    cv2.drawContours(contour_result, [ball_contour], -1, (0, 255, 0), 2)

final_result = cv2.bitwise_and(
    basketball_rgb, basketball_rgb, mask=ball_mask
)


# 6. 결과 시각화
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1), plt.imshow(basketball_rgb), plt.title("Original")

plt.subplot(2, 2, 2), plt.imshow(mask_lab, cmap="gray"), plt.title("LAB Mask")

plt.subplot(2, 2, 3), plt.imshow(mask_final, cmap="gray"), plt.title("Final Mask")

plt.subplot(2, 2, 4), plt.imshow(final_result), plt.title("Final Basketball")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()