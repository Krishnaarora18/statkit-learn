import numpy as np

class GDRegressor:
    """
    Linear Regression using Full-Batch Gradient Descent.

    This estimator minimizes the Mean Squared Error (MSE) loss using
    gradient descent updates on the entire training dataset per iteration.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each gradient descent update.

    epochs : int, default=100
        Number of full gradient descent iterations.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Learned model weights(coefficients).

    bias : float
        Learned bias term.

    loss_history_ : list of float
        Stores the MSE loss value for each epoch.

    Notes
    -----
    - Uses full-batch gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs.
    - Intended for educational use; not as efficient as closed-form.
    """

    def __init__(self, learning_rate=0.1, epochs=100):
        self.epochs = epochs
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.loss_history_ = []   # Added

    def fit(self, X_train, y_train):
        """
        Fit linear regression model using gradient descent.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Training data.

        y_train : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        self.weights = 0
        self.bias = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.weights) + self.bias

            # compute and store loss (MSE)
            loss = np.mean((y_train - y_hat) ** 2)
            self.loss_history_.append(loss)

            # gradients
            db = -2 * np.mean(y_train - y_hat)
            self.bias = self.bias - (self.lr * db)

            dw = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.weights = self.weights - (self.lr * dw)

        print(self.bias, self.weights)

        return self

    def predict(self, X_test):
        """
        Predict target values for given test samples.

        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        return np.dot(X_test, self.weights) + self.bias

