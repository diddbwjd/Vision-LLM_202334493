# Project01 - 손으로 공 튕기기 카운팅

## 1. 프로젝트 개요

**주제:** 손으로 공을 연속해서 튕기는 횟수를 실시간으로 측정하는 프로그램

카메라를 통해 초록색 공과 손을 인식하고, 공의 움직임과 손의 접촉 여부를 분석하여 연속으로 공을 튕긴 횟수를 카운팅한다.

### 주요 기능
- 공이 책상에 닿았다가 다시 손바닥에 닿으면 카운트 +1
- 손바닥에 닿지 않고 공이 튕기거나 바닥에 떨어지면 카운트 0으로 초기화
- 공의 움직임이 일정 시간 이상 멈추면 카운트 0으로 초기화
- 최초 손과 공의 접촉은 카운트하지 않음
- 실시간 카운트 결과를 영상에 표시

### 활용 가능성
공을 사용한 다른 스포츠에서도 간단한 테스트에 이용할 수 있는 기술

---

## 2. 구현 방식

### Step 1. 카메라 연결
- OpenCV의 `cv2.VideoCapture()`를 이용한 실시간 영상 입력

### Step 2. 초록색 공 검출 (초록색 색 + 공의 모양(타원))
**사용 기술:** Color Segmentation, Contour Detection

- gaussian_blur, lab,mask(색상,open,close)
- contour, contourArea

### Step 3. 공의 움직임 추적
**사용 기술:** Object Tracking

- 프레임별 공의 중심 좌표 추출
- 공의 상승 및 하강 방향 판단

### Step 4. 손 전체 검출
**사용 기술:** Hand Detection (MediaPipe)

- MediaPipe Hands를 이용하여 손의 Landmark 검출
- Landmark 좌표를 기반으로 손의 영역 구성


### Step 5. 공과 손의 관계 분석
**사용 기술:** Spatial Relationship Analysis

- `cv2.pointPolygonTest()`를 이용하여 공 중심과 손 영역의 거리 계산


### Step 6. 카운트 및 실패 감지
**사용 기술:** State-based Counting

- 최초 접촉 시 카운트하지 않음
- 공이 하강한 후 상승하여 손에 접촉하면 Count +1
- 손과 공의 접촉이 유지되는 동안 중복 카운트 방지
- 공이 손에 닿지 않은 채 다시 하강하거나 화면 밖으로 이탈하면 실패 처리
- 공의 움직임이 일정 시간 이상 멈추면 실패 처리
- 실패 시 Count = 0


