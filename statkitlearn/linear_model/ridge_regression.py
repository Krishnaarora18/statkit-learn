import numpy as np

class RidgeRegressor:

    """
    Ridge Regression (L2 Regularized Linear Regression).

    This estimator solves the regularized least squares problem:

        w = (XᵀX + αI)⁻¹ Xᵀy

    where:
    - α ≥ 0 is the regularization strength
    - I is the identity matrix (with I[0][0] = 0 to avoid penalizing bias)

    Parameters
    ----------
    alpha : float, default=3
        Regularization strength. Must be a non-negative float.
        Higher values imply stronger regularization.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.

    bias : float
        Independent term (bias) in the model.

    Notes
    -----
    - Uses the closed-form ridge regression solution.
    - The intercept term is excluded from regularization.
    - Uses explicit matrix inverse via numpy.linalg.inv().
    """

    def __init__(self,alpha=3):
        self.alpha = alpha
        self.weights=None
        self.bias =None
    def fit(self,X_train,y_train):
        """
    Fit Ridge Regression model.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Training data.

    y_train : ndarray of shape (n_samples,) or (n_samples, 1)
        Target values.

    Returns
    -------
    self : object
        Returns the fitted estimator.
    """
        X_train = np.insert(X_train, 0, 1,axis=1)
        I = np.identity(X_train.shape[1])
        I[0][0] = 0 ## To not regularise the bias term
        result = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha*I).dot(X_train.T).dot(y_train)
        self.bias = result[0]
        self.weights = result[1:]

    def predict(self,X_test):
        """
    Predict using the Ridge Regression model.

    Parameters
    ----------
    X_test : ndarray of shape (n_samples, n_features)
        Samples for which to generate predictions.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted values.
    """
        return np.dot(X_test,self.weights) + self.bias
