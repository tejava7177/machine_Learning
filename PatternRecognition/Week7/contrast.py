import numpy as np
import cv2

image = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/PatternRecognition_Images/contrast.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError('Could not read the image')

noimage = np.zeros(image.shape[:2], image.dtype)
avg = cv2.mean(image)[0] / 2.0

dst1 = cv2.scaleAdd(image, 0.5, noimage)
dst2 = cv2.scaleAdd(image, 2.0, noimage)
dst3 = cv2.addWeighted(image, 0.5, noimage, 0, avg)
dst4 = cv2.addWeighted(image, 2.0, noimage, 0, -avg)

cv2.imshow('image', image)
cv2.imshow('dst1 - decrease contrast', dst1)
cv2.imshow('dst2 - increase contrast', dst2)
cv2.imshow('dst3 - decrease contrast using average', dst3)
cv2.imshow('dst4 - increase contrast using average', dst4)

cv2.waitKey(0)
cv2.destroyAllWindows()