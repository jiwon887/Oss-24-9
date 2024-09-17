import cv2 as cv

img = cv.imread('anhyeon.jpg')

# 그레이 스케일 이미지로 변환하여 gray에 저장
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT(Scale Invariant Feature Transform) 객체를 생성
sift = cv.SIFT_create()

# 키포인트와 디스크립터를 계산하여 kp, des에 저장
# 키 포인트 = 이미지 특징점의 좌표
# 디스크립터 = 특징점에 대한 벡터
kp, des = sift.detectAndCompute(gray, None)

# 아래처럼 각각 추출 가능
# kp = sift.detect(gray, None)
# des = sift.compute(gray, kp)

# gray 이미지 위에 특징점 표시
gray = cv.drawKeypoints(gray, kp, None, flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift', gray)

k = cv.waitKey()
cv.destroyAllWindows()