import numpy as np

class LassoRegressor:
    """
    L1 penalized Linear Regression using Full-Batch Gradient Descent.

    This estimator minimizes the L1 penalized Mean Squared Error (MSE) loss using
    gradient descent updates on the entire training dataset per iteration.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each gradient descent update.

    epochs : int, default=100
        Number of full gradient descent iterations.

    alpha : float, default=0.3
        Regularization strength. Must be a non-negative float.
        Higher values imply stronger regularization.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Learned model coefficients.

    bias : float
        Learned bias term.

    Notes
    -----
    - Uses full-batch gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs and value of alpha.
    """
    def __init__(self,learning_rate=0.2,epochs=100,alpha=0.3):
        self.alpha = alpha
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

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
        n_samples, n_features = X_train.shape
        # Initialising Weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.epochs):
            ## Calculate the predicted points
            y_pred = np.dot(X_train,self.weights) + self.bias

            ## Calculate the derivatives wrt weights(dw) and bias(db)
            dw = -(1/n_samples) * np.dot(X_train.T, (y_train - y_pred))
            db = -(1/n_samples) * np.sum(y_train - y_pred)
            ## Apply L1 regularization to the weights
            dw += self.alpha * np.sign(self.weights)

            ## Update Weights and bias
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
        print(self.bias, self.weights)
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