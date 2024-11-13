import numpy as np
from svm import SVM
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.05, random_state=25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVM(lr=0.001, num_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_pred)
print(f"accuracy = {accuracy}")

def get_decision_boundary(model, X, offset=0):
    x_min_0 = np.amin(X[:, 0])
    x_max_0 = np.amax(X[:, 0])

    # On the decision boundary wTx+b = 0
    x_min_1 = (offset-(model.weights[0] * x_min_0 + model.bias)) / model.weights[1]
    x_max_1 = (offset-(model.weights[0] * x_max_0 + model.bias)) / model.weights[1]
    return [x_min_0, x_max_0], [x_min_1, x_max_1]

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
x, y = get_decision_boundary(model, X_train)
plt.plot(x, y, "b")

x, y = get_decision_boundary(model, X_train, offset=1)
plt.plot(x, y, "--r")

x, y = get_decision_boundary(model, X_train, offset=-1)
plt.plot(x, y, "--r")
plt.savefig("svm.png")
plt.show()

