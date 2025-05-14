import numpy as np
import cv2

import sys

sys.path.append("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/Week11")



from convolution import filter

# ----------------------
# Roberts 마스크 기반 차영상 처리 함수
# ----------------------
def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)

    dst1, dst2 = np.abs(dst1), np.abs(dst2)
    dst = cv2.magnitude(dst1, dst2)

    dst = np.clip(dst, 0, 255).astype('uint8')
    dst1 = np.clip(dst1, 0, 255).astype('uint8')
    dst2 = np.clip(dst2, 0, 255).astype('uint8')

    return dst, dst1, dst2

# ----------------------
# 메인 실행
# ----------------------
# 이미지 읽기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images5/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# Roberts 마스크 정의
data1 = [
    -1,  0,  0,
     0,  1,  0,
     0,  0,  0
]

data2 = [
     0,  0, -1,
     0,  1,  0,
     0,  0,  0
]

# Roberts 연산 적용
dst, dst1, dst2 = differential(image, data1, data2)

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("roberts edge", dst)
cv2.imshow("dst1 (x-direction)", dst1)
cv2.imshow("dst2 (y-direction)", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()