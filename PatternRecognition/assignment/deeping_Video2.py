import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4"
cap = cv2.VideoCapture(video_path)

# 필터 정의
blur_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러링
    blur = cv2.filter2D(frame_gray, -1, blur_kernel)

    # 샤프닝
    sharpen = cv2.filter2D(frame_gray, -1, sharpen_kernel)

    # 영상 출력
    cv2.imshow("Blur", blur)
    cv2.imshow("Sharpen", sharpen)

    # 첫 프레임에 대한 히스토그램
    if frame_num == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.hist(frame_gray.ravel(), 256, [0, 256])

        plt.subplot(1, 3, 2)
        plt.title("Blurred")
        plt.hist(blur.ravel(), 256, [0, 256])

        plt.subplot(1, 3, 3)
        plt.title("Sharpened")
        plt.hist(sharpen.ravel(), 256, [0, 256])
        plt.tight_layout()
        plt.show()

    frame_num += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()