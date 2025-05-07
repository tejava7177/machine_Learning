import numpy as np
import cv2

# 사용자 정의 히스토그램 계산 함수
def calc_histo(image, hsize, ranges=[0, 256]):  # 평탄 원소의 1차원 히스토그램 계산
    hist = np.zeros((hsize, 1), np.float32)     # 히스토그램 누적 행렬
    gap = ranges[1] / hsize                     # 계급 간격

    for i in (image / gap).flat:                # image/gap : 픽셀값을 bin 크기로 나눔
        hist[int(i)] += 1

    return hist

# 영상 읽기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images4/pixel.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상 파일 읽기 오류 발생")

hsize, ranges = [32], [0, 256]                  # 히스토그램 간격수, 값 범위
gap = ranges[1] / hsize[0]
ranges_gap = np.arange(0, ranges[1]+1, gap)

# 히스토그램 계산
hist1 = calc_histo(image, hsize[0], ranges)     # 사용자 함수
hist2 = cv2.calcHist([image], [0], None, hsize, ranges)  # OpenCV 함수
hist3, bins = np.histogram(image, ranges_gap)  # numpy 함수

# 결과 출력
print("User 함수 : \n", hist1.flatten())        # 행렬을 벡터로 변환하여 출력
print("OpenCV 함수 : \n", hist2.flatten())      # 행렬을 벡터로 변환하여 출력
print("numpy 함수 : \n", hist3.astype(float))   # 행렬을 벡터로 변환하여 출력

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()