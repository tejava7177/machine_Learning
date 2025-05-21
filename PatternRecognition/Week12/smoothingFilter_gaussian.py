import numpy as np
import cv2

# -----------------------------
# 사용자 정의 2D Gaussian 마스크 생성 함수
# -----------------------------
def getGaussianMask(ksize, sigmaX, sigmaY):
    # sigma 값이 0 이하일 경우 기본값 계산
    sigma = 0.3 * ((np.array(ksize) - 1) * 0.5 - 1) + 0.8

    if sigmaX <= 0:
        sigmaX = sigma[0]
    if sigmaY <= 0:
        sigmaY = sigma[1]

    # 커널의 중심 좌표
    u = np.array(ksize) // 2

    # x, y 방향 범위 설정
    x = np.arange(-u[0], u[0] + 1, 1)
    y = np.arange(-u[1], u[1] + 1, 1)
    x, y = np.meshgrid(x, y)

    # 2D Gaussian 수식 적용
    ratio = 1 / (2 * np.pi * sigmaX * sigmaY)
    v1 = (x**2) / (2 * sigmaX**2)
    v2 = (y**2) / (2 * sigmaY**2)
    mask = ratio * np.exp(-(v1 + v2))

    return mask / np.sum(mask)  # 정규화

# -----------------------------
# 이미지 불러오기
# -----------------------------
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/smoothing.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# -----------------------------
# 커널 크기 설정
# -----------------------------
ksize = (17, 5)

# 사용자 정의 2D Gaussian 마스크 생성
gaussian_2d = getGaussianMask(ksize, 0, 0)

# OpenCV 제공 1D 커널 생성 (가로/세로)
gaussian_1dX = cv2.getGaussianKernel(ksize[0], 0, cv2.CV_32F)
gaussian_1dY = cv2.getGaussianKernel(ksize[1], 0, cv2.CV_32F)

# -----------------------------
# 필터 적용
# -----------------------------
# 사용자 정의 2D Gaussian 필터
gauss_img1 = cv2.filter2D(image, -1, gaussian_2d)

# OpenCV GaussianBlur
gauss_img2 = cv2.GaussianBlur(image, ksize, 0)

# OpenCV 분리 필터 (1D 가로 * 세로)
gauss_img3 = cv2.sepFilter2D(image, -1, gaussian_1dX, gaussian_1dY)

# -----------------------------
# 결과 출력
# -----------------------------
titles = ['image', 'gauss_img1', 'gauss_img2', 'gauss_img3']
for t in titles:
    cv2.imshow(t, eval(t))

cv2.waitKey(0)
cv2.destroyAllWindows()