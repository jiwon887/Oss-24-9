import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegressionSGD, compute_accuracy

# 데이터 로드
mnist = fetch_openml('mnist_784', version=1, as_frame= False)
X, y = mnist["data"], mnist["target"].astype(np.int8)


# 정규화
X = StandardScaler().fit_transform(X)


#데이터 셋 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# 로지스텍 모델 초기화 및 학습
model = LogisticRegressionSGD(learning_rate=0.01, batch_size=64, max_epochs=100, patience=10)
model.fit(X_train, y_train, X_dev, y_dev)


# 테스트
test_accuracy = compute_accuracy(y_test, model.predict(X_test))
print(f'Test Accuracy: {test_accuracy:.4f}')