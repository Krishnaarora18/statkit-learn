from statkitlearn.neighbours import KNNClassifier
import numpy as np

X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [2, 2],
    [7, 6],
    [4, 4]
])

clf = KNNClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)