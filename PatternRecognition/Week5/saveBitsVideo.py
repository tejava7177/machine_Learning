import numpy as np
import cv2

image8 = cv2.imread("/Users/simjuheun/Downloads/images/read_color.jpg", cv2.IMREAD_COLOR)
if image8 is None:
    raise Exception("Could not read image")

image16 = np.uint16(image8 * (65535 / 255))
image32 = np.float32(image8 * (1.0 / 255))

print("image8 행렬의 일부 \n%s\n" % image8[10:12, 10:13])
print("image16 행렬의 일부 \n%s\n" % image16[10:12, 10:13])
print("image32 행렬의 일부 \n%s\n" % image32[10:12, 10:13])

cv2.imwrite("/image/bits/write_test_16.tif", image16)
cv2.imwrite("/image/bits/write_test_32.tif", image32)

cv2.imshow("image16", image16)
cv2.imshow("image32", image32)
cv2.waitKey(0)
cv2.destroyAllWindows()