import numpy as np
from perceptron import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Perceptron(lr=0.005,iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_pred)
print(f"accuracy = {accuracy}")

def get_decision_boundary(model, X):
    x_min_0 = np.amin(X[:, 0])
    x_max_0 = np.amax(X[:, 0])

    # On the decision boundary wTx+b = 0
    x_min_1 = -(model.weights[0] * x_min_0 + model.bias) / model.weights[1]
    x_max_1 = -(model.weights[0] * x_max_0 + model.bias) / model.weights[1]
    print(model.weights)
    print([x_min_0, x_max_0])
    print([x_min_1, x_max_1])
    return [x_min_0, x_max_0], [x_min_1, x_max_1]


plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
x, y = get_decision_boundary(model, X_test)
plt.plot(x, y)
plt.savefig("perceptron.png")
plt.show()

