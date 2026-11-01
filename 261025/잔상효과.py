import cv2
import matplotlib as mp

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    TARGET_WIDTH, TARGET_HE/g_gray.astype(np.float32) / 255.0)

def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """


    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = []

    for output_y in _:
        for output_x in _:

            for shift in range(TRAIL_LENGTH):
                # TODO: x값 shift
                input_x = 

                # TODO: shift한 픽셀을 누적
                # image = input_np * input_x
                if 0 <= input_x < width:
                    pixel_sum += image[output_y, input_x] * weights[shift]
                # TODO: 출력 배열에 누적 픽셀 저장
                output[output_y, output_x] = pixel_sum
                

    return output




















plt.imshow(..., cmap="gray")
plt.axis("off")