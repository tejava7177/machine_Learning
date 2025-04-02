import cv2
import numpy as np

switch_case = {
    ord('a'): "A pressed",
    ord('b'): "B pressed",
    0x41: "A 키 입력",
    0x42: "B 키 입력",
    63234: "← 왼쪽 화살표",
    63235: "→ 오른쪽 화살표",
    63232: "↑ 위쪽 화살표",
    63233: "↓ 아래쪽 화살표",
    27: "ESC pressed"
}

image = np.ones((200, 300), np.uint8) * 255
cv2.imshow("Keyboard Test", image)

while True:
    key = cv2.waitKeyEx(0)  # 키 입력까지 대기
    print(f"KeyCode: {key}")

    if key == 27:
        break

    print(switch_case.get(key, "Unknown key"))

cv2.destroyAllWindows()