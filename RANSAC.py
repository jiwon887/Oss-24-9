import cv2 as cv
import numpy as np

img1 = cv.imread('Gang1.jpg')
img2 = cv.imread('Gang2.jpg')


# 매칭된 이미지 크기 조정
scale_percent = 50
width1 = int(img1.shape[1] * scale_percent / 100)
height1 = int(img1.shape[0] * scale_percent / 100)
dim1 = (width1, height1)

width2 = int(img2.shape[1] * scale_percent / 100)
height2 = int(img2.shape[0] * scale_percent / 100)
dim2 = (width2, height2)

img1_resized = cv.resize(img1, dim1, interpolation=cv.INTER_AREA)
img2_resized = cv.resize(img2, dim2, interpolation=cv.INTER_AREA)
gray1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)

# sift 객체 생성, keypoint와 destence 계산
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 매쳐 객체 생성
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 디스크립터 매칭
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if(nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)


# 매칭 결과가 좋은 경우만 필터링 
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match]) #gm.queryIdx -> 첫 번째 이미지의 키포인트 인덱스
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match]) #gm.trainIdx -> 두 번째 이미지의 키포인트 인덱스
#호모그래피 행렬 계산
H,_ = cv.findHomography(points1,points2,cv.RANSAC) # H-> 행렬, _ -> 마스크 배열 ( 사용하지 않음 )


h1,w1 = img1_resized.shape[0], img1_resized.shape[1]
h2,w1 = img2_resized.shape[0], img2_resized.shape[1]

box1 = np.float32([[0,0], [0,h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(4,1,2)
box2 = cv.perspectiveTransform(box1, H)

img2_resized = cv.polylines(img2_resized, [np.int32(box2)], True, (0,255,0), 8)

img_match = np.empty((max(h1,h2), w1 + w1, 3), dtype= np.uint8)
cv.drawMatches(img1_resized, kp1, img2_resized, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('match', img_match)

k=cv.waitKey()
cv.destroyAllWindows()