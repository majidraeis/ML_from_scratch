import numpy as np

class Perceptron:
    def __init__(self, lr=0.001, iter=100):
        self.weights = None
        self.bias = None
        self.iter = iter
        self.lr = lr

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_hat = self._step_func(np.dot(X, self.weights) + self.bias)
            self.weights += self.lr * np.dot(np.transpose(X), y-y_hat)
            self.bias += self.lr * np.sum(y-y_hat)


    def predict(self, X):
        y_linear = np.dot(X, self.weights) + self.bias
        return self._step_func(y_linear)

    def _step_func(self, x):
        return np.where(x > 0, 1, 0)

