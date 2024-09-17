import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# 데이터 전처리
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델 정의 (AlexNet)
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (1, 1), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dense(100, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(test_images, test_labels))

# 모델 저장
model.save('alexnet.h5')

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# 예측
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# 정확도 계산
correct = np.sum(predicted_labels == test_labels.flatten())
total = test_labels.shape[0]
print(f'Test accuracy: {100 * correct / total:.2f} %')