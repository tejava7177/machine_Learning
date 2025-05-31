import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 영상 경로 설정
video_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4"
cap = cv2.VideoCapture(video_path)

# 결과 저장 경로 설정
result_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/result"
os.makedirs(result_path, exist_ok=True)

# 영상 속성 추출
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25  # FPS가 0일 경우 대비

# VideoWriter 객체 초기화 (저장용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
blur_writer = cv2.VideoWriter(os.path.join(result_path, "blur_output.mp4"), fourcc, fps, (width, height), isColor=True)
sharpen_writer = cv2.VideoWriter(os.path.join(result_path, "sharpen_output.mp4"), fourcc, fps, (width, height), isColor=True)

# 필터 정의
blur_kernel = np.ones((3, 3), dtype=np.float32) / 9.0  # 3x3 평균 블러 마스크
sharpen_kernel = np.array([[0, -1, 0],                 # 중심 강조 샤프닝 커널
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러 필터 적용
    blur = cv2.filter2D(frame_gray, -1, blur_kernel)

    # 샤프닝 필터 적용
    sharpen = cv2.filter2D(frame_gray, -1, sharpen_kernel)

    # 시각화용 윈도우
    cv2.imshow("Blur", blur)
    cv2.imshow("Sharpen", sharpen)

    #히스토그램 비교
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
        plt.savefig(os.path.join(result_path, "histogram_comparison.png"))
        plt.show()

    #영상 저장 (GRAY → BGR 변환 후 저장)
    blur_writer.write(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR))
    sharpen_writer.write(cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR))

    frame_num += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 🧹 종료 처리
cap.release()
blur_writer.release()
sharpen_writer.release()
cv2.destroyAllWindows()