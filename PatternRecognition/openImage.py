import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# 파일 선택 창 띄우기
root = tk.Tk()
root.withdraw()  # tkinter GUI 숨기기
file_path = filedialog.askopenfilename()  # 파일 선택

# PIL 이미지 열기 및 numpy 배열로 변환
image = Image.open(file_path).convert('RGB')
image_np = np.array(image)

# RGB → BGR 변환 (OpenCV는 BGR 사용)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# OpenCV 창으로 이미지 표시
cv2.imshow("Image", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()