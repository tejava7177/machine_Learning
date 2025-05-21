import numpy as np
import cv2

# 사용자 정의 미디언 필터 함수
def median_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2].flatten()                          # 1차원 벡터 변환
            sort_mask = cv2.sort(mask, cv2.SORT_EVERY_COLUMN)            # 오름차순 정렬
            dst[i, j] = sort_mask[sort_mask.size // 2].item()            # 중앙값 추출

    return dst

# 소금-후추 노이즈 생성 함수
def salt_pepper_noise(img, n):
    h, w = img.shape[:2]
    x = np.random.randint(0, w, n)
    y = np.random.randint(0, h, n)
    noise = img.copy()
    for (x_, y_) in zip(x, y):
        noise[y_, x_] = 0 if np.random.rand() < 0.5 else 255
    return noise

# 이미지 입력
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/median.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 소금-후추 노이즈 추가
noise = salt_pepper_noise(image, 500)

# 사용자 정의 미디언 필터 적용
med_img1 = median_filter(noise, 5)

# OpenCV 미디언 필터 적용
med_img2 = cv2.medianBlur(noise, 5)

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("noise", noise)
cv2.imshow("median - User", med_img1)
cv2.imshow("median - OpenCV", med_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()