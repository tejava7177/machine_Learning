import numpy as np
import cv2

# --------------------------
# 사용자 정의 침식 함수
def erode(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None:
        mask = np.ones((3, 3), np.uint8)
    mcnt = cv2.countNonZero(mask)
    xcenter, ycenter = int(mask.shape[1] / 2), int(mask.shape[0] / 2)

    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = img[y1:y2, x1:x2]
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp)
            dst[i, j] = 255 if cnt == mcnt else 0
    return dst

# --------------------------
# 사용자 정의 팽창 함수
def dilate(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None:
        mask = np.ones((3, 3), np.uint8)
    xcenter, ycenter = mask.shape[1] // 2, mask.shape[0] // 2

    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = img[y1:y2, x1:x2]
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp)
            dst[i, j] = 0 if cnt == 0 else 255
    return dst

# --------------------------
# 열림 연산 (침식 후 팽창)
def opening(img, mask):
    tmp = erode(img, mask)
    return dilate(tmp, mask)

# 닫힘 연산 (팽창 후 침식)
def closing(img, mask):
    tmp = dilate(img, mask)
    return erode(tmp, mask)

# --------------------------
# 영상 입력
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/morph.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 마스크 정의 (3x3 십자형)
mask = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

# 이진화 처리
th_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

# --------------------------
# 사용자 정의 연산 수행
dst_erode  = erode(th_img, mask)
dst_dilate = dilate(th_img, mask)
dst_open   = opening(th_img, mask)
dst_close  = closing(th_img, mask)

# OpenCV 연산 (참고용)
cv_erode  = cv2.erode(th_img, mask)
cv_dilate = cv2.dilate(th_img, mask)
cv_open   = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)
cv_close  = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, mask)

# --------------------------
# 결과 출력
cv2.imshow("Original", image)
cv2.imshow("Binary", th_img)
cv2.imshow("User Erode", dst_erode)
cv2.imshow("OpenCV Erode", cv_erode)
cv2.imshow("User Dilate", dst_dilate)
cv2.imshow("OpenCV Dilate", cv_dilate)
cv2.imshow("User Opening", dst_open)
cv2.imshow("OpenCV Opening", cv_open)
cv2.imshow("User Closing", dst_close)
cv2.imshow("OpenCV Closing", cv_close)

cv2.waitKey(0)
cv2.destroyAllWindows()