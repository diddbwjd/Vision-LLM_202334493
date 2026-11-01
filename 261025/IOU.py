# IOU 구현
import cv2
import numpy as np
import matplotlib.pyplot as plt
apple = cv2.imread("src/images/apple.png")
apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
def convert_bbox_coord(bbox):
    x, y, w, h = bbox

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return (x1, y1, x2, y2)

def calculate_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    
    # 각 BBox 넓이
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    
    # 교집합 영역 좌표
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = area_a + area_b - intersection_area


    # TODO: ZeroDivisionError 주의
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

#TO DO 이후 셀 코드 그대로 가져왔습니다.

apples1 = cv2.imread("src/images/two_apples1.png")
apples2 = cv2.imread("src/images/two_apples2.png")
apples3 = cv2.imread("src/images/two_apples3.png")
apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)
    
def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn

cntr_red1, cntr_grn1 = get_contour(apples1)
cntr_red2, cntr_grn2 = get_contour(apples2)
cntr_red3, cntr_grn3 = get_contour(apples3)

bbox1 = cv2.boundingRect(cntr_red1)
bbox2 = cv2.boundingRect(cntr_grn1)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples1_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples1_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

iou1 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))

bbox1 = cv2.boundingRect(cntr_red2)
bbox2 = cv2.boundingRect(cntr_grn2)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples2_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples2_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

iou2 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))

bbox1 = cv2.boundingRect(cntr_red3)
bbox2 = cv2.boundingRect(cntr_grn3)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples3_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples3_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

iou3 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb), plt.title(f"IoU: {iou1:.4f}")
plt.subplot(1,3,2), plt.imshow(apples2_rgb), plt.title(f"IoU: {iou2:.4f}")
plt.subplot(1,3,3), plt.imshow(apples3_rgb), plt.title(f"IoU: {iou3:.4f}")

for ax in plt.gcf().axes:
    ax.axis("off")