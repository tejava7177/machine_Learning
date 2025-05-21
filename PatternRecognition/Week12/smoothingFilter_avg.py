import numpy as np
import cv2

# 사용자 정의 평균 필터 함수 정의
def average_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1

            # 경계 조건 확인
            if y1 < 0 or y2 > rows or x1 < 0 or x2 > cols:
                dst[i, j] = image[i, j]  # 경계 밖은 원래 값 유지
            else:
                mask = image[y1:y2, x1:x2]       # 마스크 지정
                dst[i, j] = cv2.mean(mask)[0]    # 평균값 할당

    return dst

# 영상 입력
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images6/filter_avg.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 사용자 정의 평균 필터 적용
avg_img = average_filter(image, 5)

# OpenCV 평균 필터 및 박스 필터
blur_img = cv2.blur(image, (5, 5), anchor=(-1, -1), borderType=cv2.BORDER_REFLECT)     # 평균 필터
box_img = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))                                 # 박스 필터

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("avg_img", avg_img)
cv2.imshow("blur_img", blur_img)
cv2.imshow("box_img", box_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
