import numpy as np
import cv2

# 값이 있는 첫 번째/마지막 계급 검색 함수
def search_value_idx(hist, bias=0):
    for i in range(hist.shape[0]):
        idx = np.abs(bias - i)
        if hist[idx] > 0:
            return idx
    return -1

# 히스토그램 시각화 함수 (막대 그리기)
def draw_histo(hist, shape=(200, 360)):
    hist_img = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = shape[1] / hist.shape[0]

    for i, h in enumerate(hist):
        x = int(i * gap)
        w = int(gap)
        cv2.rectangle(hist_img, (x, 0), (x + w - 1, int(h[0])), (0, 0, 0), cv2.FILLED)

    return cv2.flip(hist_img, 0)

# 영상 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images4/hist_stretch.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 히스토그램 설정
bsize, ranges = [64], [0, 256]
hist = cv2.calcHist([image], [0], None, bsize, ranges)

# 히스토그램 기반 최소/최대 화소값 계산
bin_width = ranges[1] / bsize[0]
low = search_value_idx(hist, 0) * bin_width
high = search_value_idx(hist, bsize[0] - 1) * bin_width

# 룩업테이블(LUT) 생성: 선형 스케일링
idx = np.arange(0, 256)
idx = (idx - low) / (high - low) * 255      # 수식이 중요함.
idx[0:int(low)] = 0
idx[int(high)+1:] = 255

# LUT 적용하여 히스토그램 스트레칭 결과 생성
dst = cv2.LUT(image, idx.astype('uint8'))

# 히스토그램 재계산 및 시각화
hist_dst = cv2.calcHist([dst], [0], None, bsize, ranges)
hist_img = draw_histo(hist, (200, 360))
hist_dst_img = draw_histo(hist_dst, (200, 360))

# 출력
print("high_value =", high)
print("low_value =", low)

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.imshow("hist_img", hist_img)
cv2.imshow("hist_dst_img", hist_dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()