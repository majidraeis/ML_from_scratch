import numpy as np
class KMEANS:
    def __init__(self, num_classes, num_iters=100):
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.clusters = np.arange(num_classes)
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        centroid_init_indices= np.random.choice(n_samples, self.num_classes, replace=False)
        self.centroids = X[centroid_init_indices, :]
        for _ in range(self.num_iters):
            y = self._assign(X, self.centroids)
            new_centroids = self._get_centroids(X, y)
            if not self._calculate_dist(new_centroids, self.centroids):
                print("Reached convergence with less iterations")
                break
            self.centroids = new_centroids


    def _assign(self, X, centroids):
        preds = []
        for x in X:
            distances = [self._calculate_dist(x, centroid) for centroid in centroids]
            preds.append(self.clusters[np.argmin(distances)])
        return np.array(preds)

    def _calculate_dist(self, x, z):
        return np.sqrt(np.sum((x-z)**2))

    def _get_centroids(self, X, y):
        new_centroids = []
        for cluster in self.clusters:
            new_centroid = np.mean(X[np.argwhere(y==cluster).flatten(), :], axis=0)
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def predict(self, X):
        return self._assign(X, self.centroids)