import numpy as np
import cv2

# 최소값 & 최대값 필터링 함수
def minmax_filter(image, ksize, mode):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)  # 출력 영상 초기화
    center = ksize // 2                     # 마스크 중심 위치

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2]                   # 마스크 영역 추출
            dst[i, j] = cv2.minMaxLoc(mask)[mode]        # mode=0: 최소값, mode=1: 최대값

    return dst

# 영상 입력
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/min_max.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 최소값 필터링 (mode=0), 최대값 필터링 (mode=1)
minfilter_img = minmax_filter(image, 3, 0)
maxfilter_img = minmax_filter(image, 3, 1)


cv2.imshow("image", image)
cv2.imshow("minfilter_img", minfilter_img)
cv2.imshow("maxfilter_img", maxfilter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()