import numpy as np
import cv2

def onChanged(value):
    global image, title;
    add_value = value - int(image[0][0])
    print("추가 화소값:", add_value)
    image[:] = image + add_value
    cv2.imshow(title, image)


image = np.zeros((400, 600), np.uint8)

title = "Trackbar Event"
cv2.imshow(title, image)

cv2.createTrackbar("Brighteness", title, image[0][0], 255, onChanged)

cv2.waitKey(0)
cv2.destroyAllWindows()