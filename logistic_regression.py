import numpy as np


# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 엔트로피 손실 함수
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 정확도 계산 함수
def compute_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    return np.mean(y_true == y_pred_labels)

def shuffle_data(X, y):
    permutation = np.random.permutation(len(X))
    return X[permutation], y[permutation]




class LogisticRegressionSGD:
    
    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=100, patience=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    # 모델 학습 함수
    def fit(self, X_train, y_train, X_dev, y_dev):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        best_dev_accuracy = 0
        patience_counter = 0
        
        
        for epoch in range(self.max_epochs):
            
            # 각 epoch마다 데이터 셔플링
            X_train, y_train = shuffle_data(X_train, y_train)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]
                
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = sigmoid(linear_model)
                
                dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / len(y_batch)) * np.sum(y_predicted - y_batch)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            train_accuracy = compute_accuracy(y_train, sigmoid(np.dot(X_train, self.weights) + self.bias))
            dev_accuracy = compute_accuracy(y_dev, sigmoid(np.dot(X_dev, self.weights) + self.bias))
            
            print(f'Epoch {epoch+1}: Training Accuracy: {train_accuracy:.4f}, Dev Accuracy: {dev_accuracy:.4f}')
            
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(np.int8)

    X = StandardScaler().fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle= False)

    model = LogisticRegressionSGD(learning_rate=0.01, batch_size=64, max_epochs=100, patience=10)
    model.fit(X_train, y_train, X_dev, y_dev)
    
    test_accuracy = compute_accuracy(y_test, model.predict(X_test))
    print(f'Test Accuracy: {test_accuracy:.4f}')