import numpy as np
from kmeans import KMEANS
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=3, random_state=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KMEANS(num_iters=1000, num_classes=3)
model.fit(X_train)
y_pred = model.predict(X_train)


plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)

# plt.savefig("svm.png")
plt.show()

