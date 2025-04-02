import numpy as np
import cv2

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Left Button Down')
    elif event == cv2.EVENT_LBUTTONUP:
        print('Left Button Up')
    elif event == cv2.EVENT_RBUTTONDOWN:
        print('Right Button Down')
    elif event == cv2.EVENT_RBUTTONUP:
        print('Right Button Up')

image = np.full((200, 300), 255, np.uint8)

title1, title2 = "Mouse Event1", "Mouse Event2"

cv2.imshow(title1, image)
cv2.imshow(title2, image)

cv2.setMouseCallback(title1, onMouse)

cv2.waitKey(0)
cv2.destroyAllWindows()