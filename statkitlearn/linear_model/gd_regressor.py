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
    coef_ : ndarray of shape (n_features,)
        Learned model coefficients.

    intercept_ : float
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
        self.coef_ = None
        self.intercept_ = None
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
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            # compute and store loss (MSE)
            loss = np.mean((y_train - y_hat) ** 2)
            self.loss_history_.append(loss)

            # gradients
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

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
        return np.dot(X_test, self.coef_) + self.intercept_

