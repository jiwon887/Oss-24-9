import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils

# CIFAR-100 데이터셋 불러오기 및 전처리
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
y_train = utils.to_categorical(y_train, 100)
y_test = utils.to_categorical(y_test, 100)

# 입력 데이터 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ZFNet 모델 정의
def ZFNet(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential()

    # Conv1: 7x7, stride=2, filters=96
    model.add(layers.Conv2D(96, (7, 7), strides=2, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    # Conv2: 5x5, stride=2, filters=256
    model.add(layers.Conv2D(256, (5, 5), strides=2, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    # Conv3: 3x3, filters=384
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

    # Conv4: 3x3, filters=384
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

    # Conv5: 3x3, filters=256
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    # Fully Connected Layer 1
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Fully Connected Layer 2
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# 모델 컴파일
model = ZFNet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
