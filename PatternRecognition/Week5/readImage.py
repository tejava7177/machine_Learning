import cv2
import numpy as np


def print_matInfo(name, image):
    if image.dtype == np.uint8: mat_type = 'CV_8U'
    elif image.dtype == np.int8: mat_type = 'CV_8S'
    elif image.dtype == np.uint16: mat_type = 'CV_16U'
    elif image.dtype == np.int16: mat_type = 'CV_16S'
    elif image.dtype == np.float32: mat_type = 'CV_32F'
    elif image.dtype == np.float64: mat_type = 'CV_64F'
    else: mat_type = 'Unknown'

    nchannels = image.shape[2] if image.ndim == 3 else 1

    print("%12s : depth(%s), channels(%d) -> mat_type(%sC%d)" %
          (name, image.dtype, nchannels, mat_type, nchannels))


# 이미지 읽기 및 정보 출력
title1, title2 = 'gray2gray', 'gray2color'
gray2gray = cv2.imread("/Users/simjuheun/Downloads/images/read_gray.jpg", cv2.IMREAD_GRAYSCALE)
gray2color = cv2.imread("/Users/simjuheun/Downloads/images/read_gray.jpg", cv2.IMREAD_COLOR)

print_matInfo(title1, gray2gray)
print_matInfo(title2, gray2color)


if gray2gray is None or gray2color is None:
    raise Exception("gray2gray and gray2color are not defined")

print("행렬 좌표 (100, 100) 화소 값")
print("%s %s" % (title1, gray2gray[100, 100]))
print("%s %s" % (title2, gray2color[100, 100]))

cv2.imshow(title1, gray2gray)
cv2.imshow(title2, gray2color)
cv2.waitKey(0)
cv2.destroyAllWindows()