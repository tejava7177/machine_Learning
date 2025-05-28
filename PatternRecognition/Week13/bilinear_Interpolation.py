import numpy as np
import cv2

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


def bilinear_value(img, pt):
    x,y = np.int32(pt)

    if x >= img.shape[1] - 1: x = x - 1
    if y >= img.shape[0] - 1: y = y - 1

    P1, P2, P3, P4 = np.float32(img[y:y+2, x:x+2].flatten())


    alpha = pt[0] - x
    beta = pt[1] - y

    M1 = P1 + alpha * (P2 - P1)
    M2 = P3 + alpha * (P4 - P3)

    P = M1 + beta * (M2 - M1)

    return np.clip(P, 0, 255)



# 양선형 보간 함수
def scaling_bilinear(img, size):
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])

    dst = [[bilinear_value(img, (j / ratioX, i / ratioY))  # ✅ 튜플로 전달
            for j in range(size[0])]
            for i in range(size[1])]

    return np.array(dst, img.dtype)



# 영상 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images7/interpolation.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 에러")

# 출력 크기 지정
size = (350, 400)

# 사용자 정의 보간 수행
dst1 = scaling_bilinear(image, size)  # 사용자 정의 양선형 보간
dst2 = scaling_nearest(image, size)   # 사용자 정의 최근접 보간

# OpenCV 내장 보간 수행
dst3 = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)   # OpenCV 양선형
dst4 = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)  # OpenCV 최근접

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("User_bilinear", dst1)
cv2.imshow("User_Nearest", dst2)
cv2.imshow("OpenCV_bilinear", dst3)
cv2.imshow("OpenCV_Nearest", dst4)

cv2.waitKey(0)
cv2.destroyAllWindows()