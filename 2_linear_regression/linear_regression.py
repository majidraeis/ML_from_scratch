import numpy as np

class LinearRegression():
    def __init__(self, lr=0.01, num_iter=100):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iter):
            pred = np.dot(X, self.weights) + self.bias
            dw = np.dot(X.transpose(), y - pred)
            db = np.sum(y - pred)
            self.weights += self.lr * dw
            self.bias += self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
