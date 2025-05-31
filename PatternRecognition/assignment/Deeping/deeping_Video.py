import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 영상 파일 경로
video_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/videoSource.mp4"
cap = cv2.VideoCapture(video_path)

# 📂 저장 경로 설정
result_path = "/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/result"
os.makedirs(result_path, exist_ok=True)

# 영상 속성 얻기 (프레임 크기, FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)



# 🔧 VideoWriter 객체 생성 (그레이스케일 영상도 3채널로 저장)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID' for .avi
# ⚠️ isColor=True로 설정하고, 저장 시 3채널 영상만 쓰기
roberts_writer = cv2.VideoWriter(os.path.join(result_path, "roberts_output.mp4"), fourcc, fps, (width, height), isColor=True)
prewitt_writer = cv2.VideoWriter(os.path.join(result_path, "prewitt_output.mp4"), fourcc, fps, (width, height), isColor=True)
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

    # 저장 시 변환해서 write
    roberts_writer.write(cv2.cvtColor(rob_edge, cv2.COLOR_GRAY2BGR))
    prewitt_writer.write(cv2.cvtColor(pre_edge, cv2.COLOR_GRAY2BGR))


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
# 종료
cap.release()
roberts_writer.release()
prewitt_writer.release()
cv2.destroyAllWindows()