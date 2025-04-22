import numpy as np
import cv2

blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
image = np.zeros((400, 600, 3), np.uint8) # BGR 3채널로 컬러 윈도우를 만들겠다는 것
image[:] = (255, 255, 255) # BGR 채널 값이 255로 순수 흰색 배경 생성

pt1, pt2 = (50, 50), (250, 150)
pt3, pt4 = (400, 150), (500, 50)
roi = (50, 200, 200, 100)       #왼쪽 상단 좌표 (50, 100) / weight = 200 / height 100

cv2.line(image, pt1, pt2, red)
cv2.line(image, pt3, pt4, green, 3, cv2.LINE_AA)

cv2.rectangle(image, pt1, pt2, blue, 3, cv2.LINE_4)  # pt1 , pt2 좌표를 대각선으로 하는 직사각형 생성
cv2.rectangle(image, roi, red, 3, cv2.LINE_8)
cv2.rectangle(image, (400,200,100,100), green, cv2.FILLED)

cv2.imshow('Line & Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()