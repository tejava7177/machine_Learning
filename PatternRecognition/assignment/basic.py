import cv2
import matplotlib.pyplot as plt

#Grayscale 이미지의 히스토그램 시각화

# 이미지 불러오기 (Grayscale)
img = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/image/bird.jpg", cv2.IMREAD_GRAYSCALE)

# 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 히스토그램 시각화
plt.figure(figsize=(8,4))
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.grid(True)
plt.show()


# 밝기 및 명암 대비 조절

# 밝기 조절
bright_plus = cv2.add(img, 50)
bright_minus = cv2.subtract(img, 50)

# 명암 대비 조절
contrast_high = cv2.multiply(img, 1.5)
contrast_low = cv2.multiply(img, 0.5)

# 결과 출력
cv2.imshow("Original", img)
cv2.imshow("Bright +50", bright_plus)
cv2.imshow("Bright -50", bright_minus)
cv2.imshow("Contrast x1.5", contrast_high)
cv2.imshow("Contrast x0.5", contrast_low)
cv2.waitKey(0)
cv2.destroyAllWindows()