import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None


    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y, 1)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #Check stopping criteria
        if depth>=self.max_depth or n_labels==1 or n_samples < self.min_samples_split:
            value = Counter(y).most_common(1)[0][0]
            return Node(value=value)

        #Find the best split
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        #Create Children
        best_threshold, best_feature = self._best_split(X, y, feature_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_node = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right_node = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left_node, right_node)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        best_threshold, best_feature = None, None

        for feature_idx in feature_idxs:
            X_feature = X[:, feature_idx]
            unique_values = np.unique(X_feature)
            for threshold in unique_values:
                gain = self._information_gain(X_feature, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature_idx

        return best_threshold, best_feature

    def _information_gain(self, X_feature, y, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_feature, threshold)
        y_left, y_right = y[left_idxs], y[right_idxs]
        if len(y_left)==0 or len(y_right)==0:
            return 0
        left_entropy, right_entropy = self._entropy(y_left), self._entropy(y_right)
        weighted_avg_children_entropy = (len(y_left) * left_entropy + len(y_right) * right_entropy)/(len(y))
        return parent_entropy - weighted_avg_children_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])

    def _split(self, X, threshold):
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        return left_idxs, right_idxs


    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] > node.threshold:
            return self._predict(x, node.right)
        return self._predict(x, node.left)