import numpy as np


class SGDRegressor:
    """
    Linear Regression using Stochastic Gradient Descent (SGD).

    This estimator minimizes the Mean Squared Error (MSE) by performing
    parameter updates for each single training example. At every epoch,
    the training samples are shuffled to ensure better convergence.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each parameter update.

    epochs : int, default=100
        Number of passes over the entire dataset.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Weight vector learned by the model.

    bias : float
        Bias term.

    loss_history_ : list of float
        Stores the average MSE for each epoch.

    Notes
    -----
    - Unlike batch gradient descent, SGD updates parameters for each sample.
    - May converge faster but is noisier.
    - Works well for large datasets.
    """

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history_ = []  # Added

    def fit(self, X_train, y_train):
        """
        Fit linear regression model using stochastic gradient descent.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y_train : ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        self.bias = 0
        self.weights = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            epoch_losses = []
            idx = np.random.permutation(X_train.shape[0])

            for j in idx:
                y_hat = np.dot(X_train[j], self.weights) + self.bias

                # compute loss for this sample
                loss = (y_train[j] - y_hat) ** 2
                epoch_losses.append(loss)

                # gradients
                db = -2 * (y_train[j] - y_hat)
                self.bias = self.bias - (self.lr * db)

                dw = -2 * np.dot((y_train[j] - y_hat), X_train[j])
                self.weights = self.weights - (self.lr * dw)

            # store average epoch loss
            self.loss_history_.append(np.mean(epoch_losses))

        print(self.weights, self.bias)
        return self

    def predict(self, X_test):
        """
        Predict using the linear model.

        Parameters
        ----------
        X_test : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return np.dot(X_test, self.weights) + self.bias
