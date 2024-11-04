import numpy as np
from naive_bayes import NaiveBayes
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test)/len(y_pred)
print(f"accuracy = {accuracy}")

