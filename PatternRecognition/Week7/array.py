import cv2

image = cv2.imread('/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/bird.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise Exception('Could not read the image')


x_axis = cv2.flip(image, 0)
y_axis = cv2.flip(image, 1)
xy_axis = cv2.flip(image, -1)
rep_image = cv2.repeat(image, 2, 2)
trans_image = cv2.transpose(image)


titles = ['image', 'x_axis', 'y_axis', 'xy_axis', 'rep_image', 'trans_image']
for title in titles:
    # eval() 메서드 기억하기
    cv2.imshow(title, eval(title))



cv2.waitKey(0)
cv2.destroyAllWindows()

