import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 🔧 기능 함수 정의
# ========================================

def adjust_brightness(img, delta):
    """밝기 조절: 픽셀 값 + 또는 -"""
    return cv2.add(img, delta) if delta > 0 else cv2.subtract(img, -delta)

def adjust_contrast(img, factor):
    """명암 대비 조절: 픽셀 값 * 계수"""
    return cv2.multiply(img, factor)

def compute_histogram(img):
    """히스토그램 계산"""
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def plot_histograms(images, titles):
    """히스토그램을 한 번에 시각화"""
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        hist = compute_histogram(img)
        plt.plot(hist, label=titles[i])
    plt.title("Histogram Comparison")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

def show_images(images, titles):
    """OpenCV 윈도우로 이미지 출력"""
    for i, img in enumerate(images):
        cv2.imshow(titles[i], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========================================
# 🚀 메인 실행 로직
# ========================================

# 1. 이미지 불러오기
img = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 밝기 및 명암 조절 이미지 생성
bright_plus = adjust_brightness(img, 50)
bright_minus = adjust_brightness(img, -50)
contrast_high = adjust_contrast(img, 1.5)
contrast_low = adjust_contrast(img, 0.5)

# 3. 히스토그램 비교 시각화
images = [img, bright_plus, bright_minus, contrast_high, contrast_low]
titles = ['Original', 'Bright +50', 'Bright -50', 'Contrast x1.5', 'Contrast x0.5']
plot_histograms(images, titles)

# 4. 이미지 결과 비교 출력
show_images(images, titles)