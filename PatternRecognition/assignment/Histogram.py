import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg', cv2.IMREAD_GRAYSCALE)

# 히스토그램 스트레칭
min_val = np.min(img)
max_val = np.max(img)
stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 히스토그램 평활화
equalized = cv2.equalizeHist(img)

# 시각화
titles = ['Original', 'Stretched', 'Equalized']
images = [img, stretched, equalized]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()