NumPy를 이용한 잔상 효과 구현

- 현재 픽셀과 왼쪽 픽셀들의 값의 가중합 -> 오른쪽 방향으로 잔상이 남음
- input_x = output_x - shift : 현재 위치에서 왼쪽 픽셀의 좌표 계산
- if 문 내에서 pixel_sum += image[output_y, input_x] * weights[shift] : 
  해당 픽셀값에 weights[shift]를 곱해 pixel_sum에 누적

CuPy를 이용한 잔상 효과 구현
- image[:, :image.shape[1]-shift]로 이미지 일부 가져옴(좌표 가져올 필요 없음)
- output[:, shift:]로 픽셀값을 누적할 출력 영역을 정함
- numpy와 달리 픽셀을 하나씩 계산하지 않고, 슬라이싱을 이용 -> GPU에서 병렬 연산

MAIN GPU 구현
- torch.device("cpu") → torch.device("cuda")로 변경하여 GPU를 사용
- 기존의 .to(device)를 통해 모델과 데이터를 GPU로 이동
- torch.cuda.synchronize()를 추가하여 GPU 연산이 완료된 후 실행 시간을 측정

IoU 함수 구현
- calculate_iou() : 두 BBox의 좌표를 이용해 각 박스의 넓이 계산
- max(), min()을 이용해 두 박스가 겹치는 영역의 좌표를 구하기
- 두 BBox가 얼마나 겹치는지 0~1 사이의 값
