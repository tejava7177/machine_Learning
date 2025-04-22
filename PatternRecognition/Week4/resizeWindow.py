import cv2
import numpy as np

image = np.zeros((200, 200), np.uint8)

image[:] = 0
image

title1, title2 = "position1", "position2"
cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
cv2.moveWindow(title1, 0, 0)
cv2.moveWindow(title2, 500, 50)


cv2.imshow(title1, image)
cv2.imshow(title2, image)

cv2.resizeWindow(title1, 300, 300)
cv2.resizeWindow(title2, 400, 300)

cv2.waitKey(0)
cv2.destroyAllWindows()