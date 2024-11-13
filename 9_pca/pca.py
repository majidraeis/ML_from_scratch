import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        covariance = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvectors = eigenvectors.T
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_indices = sorted_indices[:self.n_components]
        self.components = eigenvectors[sorted_indices]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)