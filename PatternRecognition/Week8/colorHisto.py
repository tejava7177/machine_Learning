import numpy as np
import cv2

# hue 채널 기반 팔레트 생성 함수
def make_palette(rows):
    hue = [round(i * 180 / rows) for i in range(rows)]  # hue 값 리스트 계산
    hsv = [[[h, 255, 255]] for h in hue]                # HSV 색상 (V=255, S=255)
    hsv = np.array(hsv, np.uint8)                       # uint8형 변환
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)         # HSV → BGR 변환

# 색상 히스토그램 막대 이미지 생성 함수
def draw_hist_hue(hist, shape=(200, 256, 3)):
    hsv_palette = make_palette(hist.shape[0])           # 색상 팔레트 생성
    hist_img = np.full(shape, 255, np.uint8)            # 흰색 배경 이미지
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)  # 정규화

    gap = hist_img.shape[1] / hist.shape[0]             # 막대 너비 계산
    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        color = tuple(map(int, hsv_palette[i][0]))
        cv2.rectangle(hist_img, (x, 0), (x + w, int(h[0])), color, cv2.FILLED)

    return cv2.flip(hist_img, 0)  # 상하 반전 (아래가 0이 되도록)

# 이미지 로드 및 색상 히스토그램 시각화
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images4/hue_hist.jpg", cv2.IMREAD_COLOR)  # BGR 이미지 읽기
if image is None:
    raise Exception("영상파일 읽기 오류")

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)               # BGR → HSV 변환
hue_hist = cv2.calcHist([hsv_img], [0], None, [18], [0, 180])  # Hue 채널 히스토그램
hue_hist_img = draw_hist_hue(hue_hist, (200, 360, 3))          # 막대 그래프 이미지 생성

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("hue_hist_img", hue_hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()