import numpy as np
import cv2

# 사용자 정의 히스토그램 계산 함수
def calc_histo(image, hsize, ranges=[0, 256]):
    hist = np.zeros((hsize, 1), np.float32)
    gap = ranges[1] / hsize
    for i in (image / gap).flat:
        hist[int(i)] += 1
    return hist

# 히스토그램 시각화 함수
def draw_histo(hist, size=(256, 200)):
    hist_img = np.full((size[1], size[0], 3), 255, np.uint8)  # 흰 배경
    cv2.normalize(hist, hist, 0, size[1], cv2.NORM_MINMAX)    # 정규화
    gap = size[0] // hist.shape[0]

    for i in range(hist.shape[0]):
        x = i * gap
        y = size[1] - int(hist[i])
        cv2.rectangle(hist_img, (x, y), (x + gap - 1, size[1]), (0, 0, 0), cv2.FILLED)

    return hist_img

# 이미지 읽기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images4/pixel.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 히스토그램 계산
hsize, ranges = [32], [0, 256]
hist_user = calc_histo(image, hsize[0], ranges)  # 사용자 함수
hist_cv = cv2.calcHist([image], [0], None, hsize, ranges)  # OpenCV 함수

# 히스토그램 시각화
hist_img_user = draw_histo(hist_user)
hist_img_cv = draw_histo(hist_cv)

# 출력
cv2.imshow("image", image)
cv2.imshow("User Histogram", hist_img_user)

cv2.waitKey(0)
cv2.destroyAllWindows()