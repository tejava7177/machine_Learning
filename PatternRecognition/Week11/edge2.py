import numpy as np
import cv2

# 이미지 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images5/laplacian.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 4방향 Laplacian 필터
data1 = [
    [0,  1,  0],
    [1, -4,  1],
    [0,  1,  0]
]

# 8방향 Laplacian 필터
data2 = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

# 마스크 ndarray로 변환 (int16: 음수 포함)
mask4 = np.array(data1, np.int16)
mask8 = np.array(data2, np.int16)

# OpenCV 회선 함수 filter2D 적용
dst1 = cv2.filter2D(image, cv2.CV_16S, mask4)      # 4방향
dst2 = cv2.filter2D(image, cv2.CV_16S, mask8)      # 8방향
dst3 = cv2.Laplacian(image, cv2.CV_16S, ksize=1)   # OpenCV 내장 Laplacian

# 결과 정수형 변환 (절댓값 및 클램핑)
dst1 = cv2.convertScaleAbs(dst1)
dst2 = cv2.convertScaleAbs(dst2)
dst3 = cv2.convertScaleAbs(dst3)

# 결과 영상 출력
cv2.imshow("image", image)
cv2.imshow("filter2D 4-direction", dst1)
cv2.imshow("filter2D 8-direction", dst2)
cv2.imshow("laplacian_OpenCV", dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()