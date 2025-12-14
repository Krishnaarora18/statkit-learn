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

    def validate_X(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array")
        return X
    
    def validate_fit(self, X_train,y_train):
        X_train = self.validate_X(X_train)
        y_train = np.asarray(y_train)
        if y_train.ndim != 1:
            raise ValueError(f"y_train must be 1D array got {y_train.ndim}D instead")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Number of samples in X_train and y_train must be same")
        
        if not np.all(np.isfinite(X_train)):
            raise ValueError("X_train contains NaN or infinite values")

        if not np.all(np.isfinite(y_train)):
            raise ValueError("y_train contains NaN or infinite values")
        
        if not np.issubdtype(y_train.dtype, np.number):
            raise ValueError("y_train must be numeric for regression")
        
        if not isinstance(self.alpha, (int,float)) or self.alpha < 0:
                raise ValueError("alpha must be 0 or a positive real number")

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
        X_train, y_train = self.validate_fit(X_train,y_train)
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
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first")
        X_test = self.validate_X(X_test)
        X_test = np.insert(X_test, 0, 1, axis =1)
        return np.dot(X_test,self.weights) + self.bias
