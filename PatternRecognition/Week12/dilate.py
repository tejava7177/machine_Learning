import numpy as np
import cv2

# -----------------------------
# 사용자 정의 팽창 함수
# -----------------------------
def dilate(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)

    if mask is None:
        mask = np.ones((3, 3), np.uint8)  # 기본 3x3 마스크

    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0]

    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1

            roi = img[y1:y2, x1:x2]
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp)
            dst[i, j] = 0 if cnt == 0 else 255  # 하나라도 겹치면 흰색 (255)

    return dst

# -----------------------------
# 이미지 불러오기 및 이진화
# -----------------------------
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/morph.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 사용자 정의 마스크 (십자형)
mask = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

# 이진화 처리
th_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

# -----------------------------
# 팽창 연산 수행
# -----------------------------
dst1 = dilate(th_img, mask)            # 사용자 정의 함수
dst2 = cv2.dilate(th_img, mask)        # OpenCV 내장 함수
# dst2 = cv2.morphologyEx(th_img, cv2.MORPH_DILATE, mask)  # 동일 연산 (선택적)

# -----------------------------
# 결과 출력
# -----------------------------
cv2.imshow("User dilate", dst1)
cv2.imshow("OpenCV dilate", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()