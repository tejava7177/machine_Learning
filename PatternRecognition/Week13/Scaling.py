import numpy as np
import cv2
import time

# [방법1] 점방향 사상 - 점방향 배열 방식
def scaling(img, size):
    dst = np.zeros(size[::-1], img.dtype)  # 출력 이미지 초기화
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])  # Y, X 방향 축소/확대 비율

    y = np.arange(0, img.shape[0], 1)
    x = np.arange(0, img.shape[1], 1)
    y, x = np.meshgrid(y, x)

    i, j = np.int32(y * ratioY), np.int32(x * ratioX)
    dst[i, j] = img[y, x]

    return dst

# [방법2] 반복문 기반 점방향 사상
def scaling2(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            i, j = int(y * ratioY), int(x * ratioX)
            dst[i, j] = img[y, x]

    return dst

# 함수 실행 시간 체크
def time_check(func, image, size, title):
    start_time = time.perf_counter()
    ret_img = func(image, size)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(title, "수행시간 = %.2f ms" % elapsed)
    return ret_img

# 영상 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images7/scaling.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 에러")

# 축소 수행
dst1 = scaling(image, (150, 200))
dst2 = scaling2(image, (150, 200))

# 확대 + 시간 측정
dst3 = time_check(scaling, image, (300, 400), "[방법1]: 점방향배열 방식")
dst4 = time_check(scaling2, image, (300, 400), "[방법2]: 반복문 방식")

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("dst1- zoom out", dst1)
cv2.imshow("dst3- zoom out", dst3)
cv2.resizeWindow("dst1- zoom out", 260, 200)

cv2.waitKey(0)
cv2.destroyAllWindows()