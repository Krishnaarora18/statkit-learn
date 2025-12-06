from statkitlearn.linear_model import LassoRegressor
import numpy as np

def test_lasso_regression():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LassoRegressor()
    model.fit(X, y)

    pred = model.predict(np.array([[5]]))
    print("Prediction:", pred)

test_lasso_regression()