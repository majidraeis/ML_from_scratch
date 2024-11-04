import numpy as np
from collections import Counter

def calculate_dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [calculate_dist(x, z) for z in self.X]
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[1:self.k+1]
        k_nearest_ys = [self.y[ind] for ind in k_nearest_indices]
        majority_vote = Counter(k_nearest_ys).most_common()[0][0]
        return majority_vote




