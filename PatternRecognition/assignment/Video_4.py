import cv2
import numpy as np
import matplotlib.pyplot as plt

# 영상 열기
cap = cv2.VideoCapture("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4")

plt.ion()  # 실시간 업데이트를 위한 인터랙티브 모드 활성화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 영상 출력
    cv2.imshow("Enhanced Video", enhanced)

    # 히스토그램 계산
    hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])

    # 히스토그램 출력 (matplotlib)
    plt.clf()  # 이전 그래프 지우기
    plt.title("Histogram of Enhanced Frame")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.pause(0.001)  # 실시간 갱신

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()