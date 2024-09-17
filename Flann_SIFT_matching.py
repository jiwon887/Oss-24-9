import cv2 as cv
import numpy as np
import time

# 이미지 읽기 및 그레이스케일 변환
img1 = cv.imread('Gang1.jpg')

img2 = cv.imread('Gang2.jpg')

# 매칭된 이미지 크기 조정
scale_percent = 50  # 이미지 크기를 50%로 축소
width1 = int(img1.shape[1] * scale_percent / 100)
height1 = int(img1.shape[0] * scale_percent / 100)
dim1 = (width1, height1)

width2 = int(img2.shape[1] * scale_percent / 100)
height2 = int(img2.shape[0] * scale_percent / 100)
dim2 = (width2, height2)

img1_resized = cv.resize(img1, dim1, interpolation=cv.INTER_AREA)
img2_resized = cv.resize(img2, dim2, interpolation=cv.INTER_AREA)


# 그레이스케일 변환
gray1_resized = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
gray2_resized = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)

# SIFT 특징점 및 디스크립터 추출
sift = cv.SIFT_create()
kp1_resized, des1_resized = sift.detectAndCompute(gray1_resized, None)
kp2_resized, des2_resized = sift.detectAndCompute(gray2_resized, None)

# 매칭 결과 필터링
T = 0.7
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match_resized = flann_matcher.knnMatch(des1_resized, des2_resized, 2)
good_match_resized = []
for nearest1, nearest2 in knn_match_resized:
    if (nearest1.distance / nearest2.distance) < T:
        good_match_resized.append(nearest1)

# 매칭 결과 그리기
img_match_resized = np.empty((max(img1_resized.shape[0], img2_resized.shape[0]),
                              img1_resized.shape[1] + img2_resized.shape[1], 3), dtype=np.uint8)

cv.drawMatches(img1_resized, kp1_resized, img2_resized, kp2_resized, good_match_resized, img_match_resized,
               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Good Matches', img_match_resized)

k = cv.waitKey()
cv.destroyAllWindows()
