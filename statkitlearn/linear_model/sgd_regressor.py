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
        self.tol = 1e-4 
        self.n_iter_no_change = 10 
        self.loss_history_ = []

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
            raise ValueError(f"Number of samples in X_train, and y_train must be same")
        
        if not isinstance(self.lr, (int,float)) or self.lr <= 0:
            raise ValueError("Learning rate must be a positive integer")
        
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Epochs must be a positive integer")
        
        if not np.all(np.isfinite(X_train)):
            raise ValueError("X_train contains NaN or infinite values")

        if not np.all(np.isfinite(y_train)):
            raise ValueError("y_train contains NaN or infinite values")
        
        if not np.issubdtype(y_train.dtype, np.number):
            raise ValueError("y_train must be numeric for regression")

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
        X_train, y_train = self.validate_fit(X_train,y_train)

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

                dw = -2 * np.dot((y_train[j] - y_hat), X_train[j])
                db = - 2 * np.mean(y_train[j] - y_hat)
                self.weights -= (self.lr * dw)
                self.bias -= self.lr * db
            # store average epoch loss
            self.loss_history_.append(np.mean(epoch_losses))

            if i > self.n_iter_no_change:
                    recent_losses = self.loss_history_[-self.n_iter_no_change:]
                    loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
                    if abs(loss_improvement) < self.tol:
                        print(f"Converged at epoch {i}")
                        break

        print(self.bias, self.weights)
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
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_test = self.validate_X(X_test)
        return np.dot(X_test, self.weights) + self.bias
