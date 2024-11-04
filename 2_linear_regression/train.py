import numpy as np
from linear_regression import LinearRegression
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_samples=200, n_features=1, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(num_iter=500)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = np.sum((y_test-pred)**2)/len(y_test)
print(f"MSE = {mse}")

plt.figure()
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, pred, color='red')
plt.show()





