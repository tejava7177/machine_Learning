import cv2
import matplotlib.pyplot as plt


# 📌 Grayscale Histogram
# 	•	cv2.calcHist()는 픽셀 빈도를 나타내는 1D 배열을 반환하며, 이는 이미지의 밝기 분포를 직관적으로 보여준다.
# 	•	어두운 이미지일수록 왼쪽에 몰려 있고, 밝은 이미지일수록 오른쪽에 밀집됨.
#
# 📌 밝기 & 명암 조절
# 	•	밝기 조절은 전체 픽셀 값에 +β를 적용하는 것
# 	•	명암 조절은 픽셀의 변동 폭을 키우거나 줄이는 것 (중간 값 기준 확대/축소)
# 	•	cv2.add, cv2.multiply는 클리핑 처리를 자동으로 하기 때문에 안정적


#Grayscale 이미지의 히스토그램 시각화

# 이미지 불러오기 (Grayscale)
img = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg", cv2.IMREAD_GRAYSCALE)

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