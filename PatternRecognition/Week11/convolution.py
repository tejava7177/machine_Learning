import numpy as np
import cv2

# 회선 수행 함수 – 행렬 처리 방식 (속도 효율적)
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)  # 회선 결과 저장 행렬
    ycenter, xcenter = mask.shape[0] // 2, mask.shape[1] // 2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):         # 입력 행렬 반복
        for j in range(xcenter, cols - xcenter):     # 입력 열 반복
            y1, y2 = i - ycenter, i + ycenter + 1    # 관심 영역 y 범위
            x1, x2 = j - xcenter, j + xcenter + 1    # 관심 영역 x 범위
            roi = image[y1:y2, x1:x2].astype('float32')   # 관심 영역 행렬
            tmp = cv2.multiply(roi, mask)                # 원소 곱셈
            dst[i, j] = cv2.sumElems(tmp)[0]             # 합산하여 저장

    return dst

# 회선 수행 함수 – 직접 곱셈 순회 (직관적이나 속도 느림)
def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)  # 결과 저장
    ycenter, xcenter = mask.shape[0] // 2, mask.shape[1] // 2

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            sum = 0.0
            for u in range(mask.shape[0]):
                for v in range(mask.shape[1]):
                    y = i + u - ycenter
                    x = j + v - xcenter
                    sum += image[y, x] * mask[u, v]   # 회선 수식
            dst[i, j] = sum

    return dst


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