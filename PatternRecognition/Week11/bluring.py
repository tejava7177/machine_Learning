import numpy as np
import cv2
import sys

sys.path.append("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/Week11")



from convolution import filter, filter2


# 영상 읽기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images5/filter_blur.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 블러링 마스크 생성 (3x3 평균 필터)
data = [
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9
]
mask = np.array(data, np.float32).reshape(3, 3)

# 회선 수행
blur1 = filter(image, mask)     # 방법 1 - 행렬 기반 회선
blur2 = filter2(image, mask)    # 방법 2 - 픽셀 직접 연산

# 결과 영상 정수형으로 변환
blur1 = blur1.astype('uint8')                  # 단순 형변환
blur2 = cv2.convertScaleAbs(blur2)             # 절댓값 및 스케일 조정

# 결과 출력
cv2.imshow("image", image)        # 원본 영상
cv2.imshow("blur1", blur1)        # 행렬 기반 결과
cv2.imshow("blur2", blur2)        # 직접 곱셈 방식 결과

cv2.waitKey(0)
cv2.destroyAllWindows()