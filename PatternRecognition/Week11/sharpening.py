import numpy as np
import cv2
import sys

sys.path.append("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/Week11")

from convolution import filter


# 영상 읽기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images5/filter_blur.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")


# 샤프닝 마스크 정의
data1 = [
    0, -1,  0,
   -1,  5, -1,
    0, -1,  0
]

data2 = [
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
]

# 마스크 배열로 변환
mask1 = np.array(data1, np.float32).reshape(3, 3)
mask2 = np.array(data2, np.float32)

# 회선 수행
sharpen1 = filter(image, mask1)
sharpen2 = filter(image, mask2)

# 결과 정수형으로 변환 (디스플레이를 위해)
sharpen1 = cv2.convertScaleAbs(sharpen1)
sharpen2 = cv2.convertScaleAbs(sharpen2)

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("sharpen1", sharpen1)
cv2.imshow("sharpen2", sharpen2)

cv2.waitKey(0)
cv2.destroyAllWindows()