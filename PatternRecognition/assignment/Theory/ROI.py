import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_stretching(roi):
    min_val = np.min(roi)
    max_val = np.max(roi)
    stretched = (roi - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)

# 이미지 불러오기
img = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

# ROI 지정 (예시: 중앙 100x100)
h, w = img.shape
x, y = w // 2 - 50, h // 2 - 50
roi = img[y:y+100, x:x+100]

# 처리
equalized_roi = cv2.equalizeHist(roi)
stretched_roi = histogram_stretching(roi)

# 시각화를 위한 배치
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

titles = ['Original ROI', 'Equalized ROI', 'Stretched ROI']
images = [roi, equalized_roi, stretched_roi]

for i in range(3):
    axs[i][0].imshow(images[i], cmap='gray')
    axs[i][0].set_title(titles[i])
    axs[i][0].axis('off')

    axs[i][1].hist(images[i].ravel(), bins=256, range=[0, 256])
    axs[i][1].set_title(f"{titles[i]} Histogram")

plt.tight_layout()
plt.show()