import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/assignment/source/bird.jpg', cv2.IMREAD_GRAYSCALE)

# 로버츠
def apply_roberts(img):
    kx = np.array([[1, 0], [0, -1]])
    ky = np.array([[0, 1], [-1, 0]])
    gx = cv2.filter2D(img, -1, kx)
    gy = cv2.filter2D(img, -1, ky)
    return cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))

# 프리윗
def apply_prewitt(img):
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    gx = cv2.filter2D(img, -1, kx)
    gy = cv2.filter2D(img, -1, ky)
    return cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))

# 히스토그램
def plot_histogram(image, title="Histogram"):
    plt.figure()
    plt.title(title)
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.grid()
    plt.show()

# 블러 / 샤프닝
blur_k = np.ones((3,3), np.float32)/9
sharp_k = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

blurred = cv2.filter2D(img, -1, blur_k)
sharpened = cv2.filter2D(img, -1, sharp_k)

# 출력
cv2.imshow('Original', img)
cv2.imshow('Blurred', blurred)
cv2.imshow('Sharpened', sharpened)
cv2.imshow('Roberts', apply_roberts(img).astype(np.uint8))
cv2.imshow('Prewitt', apply_prewitt(img).astype(np.uint8))

plot_histogram(img, "Original")
plot_histogram(blurred, "Blurred")
plot_histogram(sharpened, "Sharpened")

cv2.waitKey(0)
cv2.destroyAllWindows()