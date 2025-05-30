import cv2
import numpy as np

# 영상 저장 설정
width, height = 256, 256
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/test_video2.mp4', fourcc, 30.0, (width, height))

# 30프레임짜리 단순한 영상 생성
for i in range(30):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # 흰 배경
    top_left = (10 + i * 5, 100)
    bottom_right = (60 + i * 5, 150)
    #cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)  # 검은 사각형
    #cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)  # 두께 2로 외곽선만
    cv2.rectangle(frame, top_left, bottom_right, (128, 128, 128), -1)
    out.write(frame)

out.release()
print("✅ test_video.mp4 생성 완료!")