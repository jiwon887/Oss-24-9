import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from MLP import MLP

# 데이터 로드
mnist = fetch_openml('mnist_784')
X = mnist.data.T
y = mnist.target.astype(int)
y = np.array(y)

# onehotencoding 벡터로 변환
scaler = StandardScaler()
X = scaler.fit_transform(X.T).T
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1)).T

# 데이터 셋 분할
X_train, X_temp, Y_train, Y_temp = train_test_split(X.T, Y.T, test_size=0.3, random_state=1)
X_dev, X_test, Y_dev, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1)

X_train, Y_train = X_train.T, Y_train.T
X_dev, Y_dev = X_dev.T, Y_dev.T
X_test, Y_test = X_test.T, Y_test.T

# 초기화
layer_dims = [784, 128, 64, 32, 10]
mlp = MLP(layer_dims=layer_dims, learning_rate=0.01, max_epochs=100, minibatch_size=64, early_stopping_rounds=10)

# 학습 
mlp.fit(X_train, Y_train, X_dev, Y_dev)

# 비용 , 정확도 계산
AL_test, _ = mlp.forward_propagation(X_test)
test_cost = mlp.compute_cost(AL_test, Y_test)
test_accuracy = mlp.compute_accuracy(AL_test, Y_test)

print(f"Test Cost: {test_cost:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 저장
mlp.save_model("mlp_model.pkl")

# 모델 로드, 테스트 데이터로 평가
mlp_loaded = MLP(layer_dims=layer_dims)
mlp_loaded.load_model("mlp_model.pkl")

AL_test_loaded, _ = mlp_loaded.forward_propagation(X_test)
test_accuracy_loaded = mlp_loaded.compute_accuracy(AL_test_loaded, Y_test)

print(f"Test Accuracy (Loaded Model): {test_accuracy_loaded:.4f}")
