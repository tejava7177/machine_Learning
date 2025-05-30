import cv2
import numpy as np
import matplotlib.pyplot as plt

# 영상 파일 경로
video_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4"
cap = cv2.VideoCapture(video_path)

# 로버츠 마스크
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

# 프리윗 마스크
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 로버츠 필터
    rob_x = cv2.filter2D(frame_gray, -1, roberts_x)
    rob_y = cv2.filter2D(frame_gray, -1, roberts_y)
    rob_edge = cv2.addWeighted(rob_x, 0.5, rob_y, 0.5, 0)

    # 프리윗 필터
    pre_x = cv2.filter2D(frame_gray, -1, prewitt_x)
    pre_y = cv2.filter2D(frame_gray, -1, prewitt_y)
    pre_edge = cv2.addWeighted(pre_x, 0.5, pre_y, 0.5, 0)

    cv2.imshow("Roberts", rob_edge)
    cv2.imshow("Prewitt", pre_edge)

    # 히스토그램 시각화
    if frame_num == 0:  # 처음 프레임에서만 저장
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Histogram")
        plt.hist(frame_gray.ravel(), 256, [0, 256])

        plt.subplot(1, 3, 2)
        plt.title("Roberts Histogram")
        plt.hist(rob_edge.ravel(), 256, [0, 256])

        plt.subplot(1, 3, 3)
        plt.title("Prewitt Histogram")
        plt.hist(pre_edge.ravel(), 256, [0, 256])
        plt.tight_layout()
        plt.show()
    frame_num += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()