import numpy as np
import cv2

# 사용자 정의 침식 함수
def erode(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)

    # 마스크가 없을 경우 기본 3x3 ones 생성
    if mask is None:
        mask = np.ones((3, 3), np.uint8)

    # 마스크 중심 좌표 계산
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0]

    # 마스크 내 1의 개수 계산
    mcnt = cv2.countNonZero(mask)

    # 침식 수행
    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1

            roi = img[y1:y2, x1:x2]                     # 마스크 영역 추출
            temp = cv2.bitwise_and(roi, mask)          # 마스크와 AND 연산
            cnt = cv2.countNonZero(temp)               # 일치하는 1의 개수

            dst[i, j] = 255 if cnt == mcnt else 0      # 완전히 일치하면 255

    return dst

# ----------------------------
# 이미지 입력 및 이진화
# ----------------------------
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/morph.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 사용자 정의 마스크 (3x3 십자형)
data = [
    0, 1, 0,
    1, 1, 1,
    0, 1, 0
]
mask = np.array(data, np.uint8).reshape(3, 3)

# 이진화 처리
th_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

# 사용자 정의 침식
dst1 = erode(th_img, mask)

# OpenCV 침식 연산
dst2 = cv2.erode(th_img, mask)

# ----------------------------
# 결과 출력
# ----------------------------
cv2.imshow("image", image)
cv2.imshow("binary image", th_img)
cv2.imshow("User erode", dst1)
cv2.imshow("OpenCV erode", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()