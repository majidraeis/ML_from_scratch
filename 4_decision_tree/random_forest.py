from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, num_trees=10, max_dept=100, min_samples_split=2):
        self.num_trees = num_trees
        self.max_depth = max_dept
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.num_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.transpose(tree_preds)
        # tree_preds = np.swapaxes(tree_preds, 0, 1)
        preds = np.array([self._most_common(pred) for pred in tree_preds])
        return preds

    def _most_common(self, z):
        return Counter(z).most_common(1)[0][0]

