import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout

# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 원-핫 인코딩
y_test = tf.keras.utils.to_categorical(y_test, 10)

# LeNet-5 모델 생성
# 원래 sigmoid를 사용해야하지만 그냥 relu 처박아버림
# 원래 F6에서 tanh 사용해야 하지만 그냥 softmax 처박음
# 원래 손실 함수 MSE 써야하지만 아담 박아버림
# 요즘은 MZ식으로 대충 만들기
lenet = Sequential()

# C1 - 첫 번째 합성곱 층
lenet.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))

# S2 - 첫 번째 평균 풀링 층
lenet.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C3 - 두 번째 합성곱 층
lenet.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))

# S4 - 두 번째 평균 풀링 층
lenet.add(AveragePooling2D(pool_size=(2, 2), strides=2))

# C5 - 세 번째 합성곱 층 (Fully Connected로 처리됨)
lenet.add(Conv2D(120, (5, 5), padding='valid', activation='relu'))

# Flatten - 평탄화 층
lenet.add(Flatten())

# F6 - 첫 번째 완전 연결 층
lenet.add(Dense(84, activation='relu'))

# Output layer - 출력 층
lenet.add(Dense(10, activation='softmax'))

# 모델 컴파일
# 원래는 loss = mse, optimizer = SGD를 사용하는데 이게 mz식
lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
lenet.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=1)

# 모델 평가
res = lenet.evaluate(x_test, y_test, verbose=1)
print('accuracy : ', res[1] * 100)
