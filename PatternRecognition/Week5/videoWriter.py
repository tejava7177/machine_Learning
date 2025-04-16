import cv2

def put_string(frame, text, pt, value, color=(0, 0, 255)):
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0,0,0), 2)
    cv2.putText(frame, text, pt, font, 0.7, color, 2)


def zoom_bar(value):
    global capture
    capture.set(cv2.CAP_PROP_ZOOM, value)

def focus_bar(value):
    global capture
    capture.set(cv2.CAP_PROP_FOCUS, value)


capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    raise Exception("카메라 연결 안 됨")

# print("너비 %d" % capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# print("높이 %d" % capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("노출 %d" % capture.get(cv2.CAP_PROP_EXPOSURE))
# print("밝기 %d" % capture.get(cv2.CAP_PROP_BRIGHTNESS))


title = "Change Camera Properties"
cv2.namedWindow(title)
cv2.createTrackbar('zoom', title, 0, 10, zoom_bar)
cv2.createTrackbar('focus', title, 0, 40, focus_bar)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    if cv2.waitKey(30) >= 0:
        break


    zoom = int(capture.get(cv2.CAP_PROP_ZOOM))
    focus = int(capture.get(cv2.CAP_PROP_FOCUS))

    put_string(frame, 'zoom : ', (10, 240), zoom)
    put_string(frame, 'focus : ', (10, 270), focus)

    cv2.imshow(title, frame)


capture.release()
cv2.destroyAllWindows()
