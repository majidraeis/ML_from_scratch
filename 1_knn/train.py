import numpy as np
from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model = KNN(3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_pred)
print(f"accuracy = {accuracy}")


plt.figure()
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test)
plt.title("Original clusters")

plt.figure()
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred)
plt.title("Predicted clusters")
plt.show()

