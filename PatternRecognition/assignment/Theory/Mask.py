import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg", cv2.IMREAD_GRAYSCALE)

# 블러링 마스크 (평균값 필터)
blur_kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# 샤프닝 마스크
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# 회선 적용
blurred = cv2.filter2D(img, -1, blur_kernel)
sharpened = cv2.filter2D(img, -1, sharpen_kernel)

# 출력
cv2.imshow("Original", img)
cv2.imshow("Blurred", blurred)
cv2.imshow("Sharpened", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()