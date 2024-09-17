import numpy as np
import pickle

class MLP:
    def __init__(self, layer_dims, learning_rate=0.01, max_epochs=100, minibatch_size=32, early_stopping_rounds=10):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.minibatch_size = minibatch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.parameters = self.initialize_parameters()


    # 가중치, 편향 초기화
    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    # 활성화 함수
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    # 비용 함수 계산
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return np.squeeze(cost)

    # 순전파 구현
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = self.relu(Z)
        elif activation == "softmax":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = self.softmax(Z)

        cache = (linear_cache, Z)
        return A, cache

    def forward_propagation(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="softmax")
        caches.append(cache)

        return AL, caches

    # 역전파 구현
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, Z = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, Z)
        elif activation == "softmax":
            dZ = dA

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = AL - Y

        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation="softmax")

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # 역전파에서 계산한 가중치, 편향 업데이트   
    def update_parameters(self, grads):
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]

    # 모델 학습
    def fit(self, X_train, Y_train, X_dev, Y_dev):
        np.random.seed(1)
        m = X_train.shape[1]
        dev_costs = []
        best_dev_cost = float('inf')
        best_params = self.parameters
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            for i in range(0, m, self.minibatch_size):
                X_mini = X_train_shuffled[:, i:i+self.minibatch_size]
                Y_mini = Y_train_shuffled[:, i:i+self.minibatch_size]

                AL, caches = self.forward_propagation(X_mini)
                grads = self.backward_propagation(AL, Y_mini, caches)
                self.update_parameters(grads)

            AL_train, _ = self.forward_propagation(X_train)
            train_cost = self.compute_cost(AL_train, Y_train)
            train_accuracy = self.compute_accuracy(AL_train, Y_train)

            AL_dev, _ = self.forward_propagation(X_dev)
            dev_cost = self.compute_cost(AL_dev, Y_dev)
            dev_accuracy = self.compute_accuracy(AL_dev, Y_dev)

            dev_costs.append(dev_cost)

            if dev_cost < best_dev_cost:
                best_dev_cost = dev_cost
                best_params = self.parameters
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch + 1}/{self.max_epochs}, Train Cost: {train_cost:.4f}, Train Accuracy: {train_accuracy:.4f}, Dev Cost: {dev_cost:.4f}, Dev Accuracy: {dev_accuracy:.4f}")

            if epochs_without_improvement >= self.early_stopping_rounds:
                print("Early stop")
                break

        self.parameters = best_params


    # 정확도 계산
    def compute_accuracy(self, AL, Y):
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.parameters, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            self.parameters = pickle.load(file)
