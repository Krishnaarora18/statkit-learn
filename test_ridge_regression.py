from statkitlearn.linear_model import RidgeRegressor
import numpy as np

# def test_ridge_regression():
#     X = np.array([[1], [2], [3], [4]])
#     y = np.array([2, 4, 6, 8])

#     model = RidgeRegressor()
#     model.fit(X, y)

#     pred = model.predict(np.array([[5]]))
#     print("Prediction:", pred)

# test_ridge_regression()
from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RidgeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))