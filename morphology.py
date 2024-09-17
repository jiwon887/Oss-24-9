import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('anhyeon.jpg', cv.IMREAD_UNCHANGED)

if img is None : 
    exit(1)

# png파일은 4개의 채널이 존재, 3번 채널에 그림이 있음 [:,:,3]으로 3번째 채널 저장
# THRESH_BINARY -> 고정된 임계값을 사용하여 이진화
# THRESH_OTSU -> 이미지 히스토그램에 따라 최적의 임계값을 자동으로 찾아서 이진화
#t, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#plt.imshow(bin_img , cmap= 'gray'), plt.xticks([]), plt.yticks([])
#plt.show()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
t, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

plt.imshow(bin_img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()


# 사용할 부분 자르기
b = bin_img[bin_img.shape[0] // 2 : bin_img.shape[0], 0 : bin_img.shape[0] // 2 + 1]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se=np.uint8([[0,0,1,0,0],
             [0,1,1,0,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
             [0,0,1,0,0]])


# 팽창
b_dilation = cv.dilate(b, se, iterations= 1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 침식
b_erosion = cv.erode(b, se, iterations= 1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 닫힘
b_closing = cv.erode(cv.dilate(b, se, iterations= 1), se, iterations= 1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()