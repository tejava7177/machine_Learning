import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# -- 원본 이미지 불러오기 --
img = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/image/bird.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception("이미지를 불러올 수 없습니다.")

adjusted_img = img.copy()

# -- 조절 함수 --
def apply_adjustment(alpha, beta):
    global adjusted_img
    adjusted = img.astype(np.float32) * alpha + beta
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    adjusted_img = adjusted
    cv2.imshow("Adjusted Image", adjusted_img)

# -- 슬라이더 값이 바뀔 때 호출 --
def on_change(val=None):
    alpha = alpha_slider.get() / 100.0
    beta = beta_slider.get() - 100
    apply_adjustment(alpha, beta)

# -- 버튼 클릭: 히스토그램 출력 --
def show_histogram():
    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_adj = cv2.calcHist([adjusted_img], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Histogram")
    plt.plot(hist_orig, color='gray')
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.title("Adjusted Histogram")
    plt.plot(hist_adj, color='black')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

# -- GUI 생성 --
root = tk.Tk()
root.title("Real GUI: Brightness & Contrast")

# 밝기 슬라이더
tk.Label(root, text="Alpha x100 (contrast)").pack()
alpha_slider = tk.Scale(root, from_=50, to=300, orient=tk.HORIZONTAL, command=on_change)
alpha_slider.set(100)
alpha_slider.pack(fill=tk.X)

# 명암 슬라이더
tk.Label(root, text="Beta + (brightness)").pack()
beta_slider = tk.Scale(root, from_=0, to=200, orient=tk.HORIZONTAL, command=on_change)
beta_slider.set(100)
beta_slider.pack(fill=tk.X)

# 히스토그램 버튼
tk.Button(root, text="Show Histogram", command=show_histogram).pack(pady=10)

# -- 초기 출력 --
cv2.imshow("Adjusted Image", img)

# -- GUI 루프 실행 --
root.mainloop()
cv2.destroyAllWindows()