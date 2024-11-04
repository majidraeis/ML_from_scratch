import numpy as np
from decision_tree import DecisionTree
from random_forest import RandomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split

feature_label = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(feature_label.data, feature_label.target, test_size=0.2)

# model = DecisionTree(max_depth=50)
model = RandomForest()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_pred)
print(f"accuracy = {accuracy}")

