import cv2

image = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/bird.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise Exception('Could not read the image')
if image.ndim != 3:
    raise Exception('Image must have 3 dimensions')


bgr = cv2.split(image)

print("bgr 자료형:", type(bgr), type(bgr[0]), type(bgr[0][0]), type(bgr[0][0][0]))
print("bgr 원소개수:", len(bgr))

cv2.imshow('image', image)
cv2.imshow('Blue Channel', bgr[0])
cv2.imshow('Green Channel', bgr[1])
cv2.imshow('Red Channel', bgr[2])




cv2.waitKey(0)
cv2.destroyAllWindows()