import cv2

image = cv2.imread('/Users/simjuheun/Downloads/images/read_color.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise Exception("영상 파일 읽기 에러")

params_jpg = (cv2.IMWRITE_JPEG_QUALITY, 10)
params_png = (cv2.IMWRITE_PNG_COMPRESSION, 9)

cv2.imwrite('image/test1.jpg', image)
cv2.imwrite('image/test2.jpg', image, params_jpg)
cv2.imwrite('image/test3.png', image, params_png)
cv2.imwrite('image/test4.bmp', image)

print("저장 완료")