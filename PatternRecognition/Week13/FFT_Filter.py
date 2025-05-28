import numpy as np
import cv2

def FFT_OpenCV(image):
    f_img = np.float32(image)
    dft = cv2.dft(f_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # 중심 이동

    magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    spectrum = 20 * np.log(magnitude + 1)
    spectrum = cv2.convertScaleAbs(spectrum)

    return dft_shifted, spectrum


def IFFT_OpenCV(dft_shifted, shape):
    dft_ishift = np.fft.ifftshift(dft_shifted)  # 역 중심 이동
    idft = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    idft_crop = idft[:shape[0], :shape[1]]  # padding 제거

    return cv2.convertScaleAbs(idft_crop)


# 영상 불러오기
image = cv2.imread("/Users/simjuheun/Desktop/myProject/machine_Learning/PatternRecognition/image/images7/filter.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 에러")

# FFT 수행
dft, spectrum = FFT_OpenCV(image)

# 필터 마스크 생성
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

lowpass = np.zeros((rows, cols, 2), np.float32)
highpass = np.ones((rows, cols, 2), np.float32)

cv2.circle(lowpass, (ccol, crow), 30, (1, 1), -1)   # 저주파 영역만 통과
cv2.circle(highpass, (ccol, crow), 30, (0, 0), -1)  # 고주파 영역만 통과

# 필터 적용
lowpassed_dft = dft * lowpass
highpassed_dft = dft * highpass

# 역변환
lowpassed_img = IFFT_OpenCV(lowpassed_dft, image.shape)
highpassed_img = IFFT_OpenCV(highpassed_dft, image.shape)

# 스펙트럼 시각화
spectrum_low = 20 * np.log(cv2.magnitude(lowpassed_dft[:, :, 0], lowpassed_dft[:, :, 1]) + 1)
spectrum_high = 20 * np.log(cv2.magnitude(highpassed_dft[:, :, 0], highpassed_dft[:, :, 1]) + 1)

spectrum_low = cv2.convertScaleAbs(spectrum_low)
spectrum_high = cv2.convertScaleAbs(spectrum_high)

# 결과 출력
cv2.imshow("Input Image", image)
cv2.imshow("Low-Pass Image", lowpassed_img)
cv2.imshow("High-Pass Image", highpassed_img)
cv2.imshow("Original Spectrum", spectrum)
cv2.imshow("Low-Pass Spectrum", spectrum_low)
cv2.imshow("High-Pass Spectrum", spectrum_high)

cv2.waitKey(0)
cv2.destroyAllWindows()