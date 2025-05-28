import numpy as np
import cv2

# 정방향 사상 기반 크기 변경 함수
def scaling(img, size):
    dst = np.zeros(size[::-1], img.dtype)  # 출력 이미지 초기화
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])  # Y, X 방향 축소/확대 비율

    y = np.arange(0, img.shape[0], 1)
    x = np.arange(0, img.shape[1], 1)
    y, x = np.meshgrid(y, x)

    i, j = np.int32(y * ratioY), np.int32(x * ratioX)
    dst[i, j] = img[y, x]

    return dst

def scaling_nearest(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])

    i = np.arange(0, size[1], 1)  # 출력 영상 가로 좌표
    j = np.arange(0, size[0], 1)  # 출력 영상 세로 좌표
    j, i = np.meshgrid(j, i)

    y = np.int32(j / ratioY)
    x = np.int32(i / ratioX)

    # 인덱스가 범위를 초과하지 않도록 clip으로 보정
    y = np.clip(y, 0, img.shape[0] - 1)
    x = np.clip(x, 0, img.shape[1] - 1)

    dst[i, j] = img[y, x]
    return dst

# 영상 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images7/interpolation.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 에러")

# 정방향 사상 기반 크기 변경
dst1 = scaling(image, (350, 400))

# 최근접 이웃 보간 기반 크기 변경
dst2 = scaling_nearest(image, (350, 400))

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("dst1- forward mapping", dst1)
cv2.imshow("dst2- NN interpolation", dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()