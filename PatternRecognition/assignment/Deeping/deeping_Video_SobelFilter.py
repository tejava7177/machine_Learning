import cv2
import numpy as np
import matplotlib.pyplot as plt

# 영상 불러오기
cap = cv2.VideoCapture('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sobel X, Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    # 결과 시각화
    cv2.imshow('Original', gray)
    cv2.imshow('Sobel Edge', sobel_mag)

    # 히스토그램 시각화
    hist = cv2.calcHist([sobel_mag], [0], None, [256], [0, 256])
    plt.clf()
    plt.title('Sobel Histogram')
    plt.plot(hist)
    plt.pause(0.001)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()