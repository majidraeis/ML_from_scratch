import numpy as np

class SVM:
    def __init__(self, lr=0.01, num_iter=100, lambda_param=0.01):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            preds = np.dot(X, self.weights) + self.bias
            y_pred_mults = y_ * preds
            for i, y_pred_mult in enumerate(y_pred_mults):
                if y_pred_mult >= 1:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - y_[i] * X[i])
                    self.bias += self.lr * (y_[i])


    def predict(self, X):
        y_linear = np.dot(X, self.weights) + self.bias
        return np.where(y_linear > 0, 1, 0)
