import numpy as np

class LogisticRegression:
    """
    Logistic Regression using Gradient Descent.

    This estimator minimizes the log loss error using
    gradient descent updates on the entire training dataset per iteration.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each gradient descent update.

    epochs : int, default=1000
        Number of full gradient descent iterations.

    Attributes
    ----------
    weights : ndarray of shape (n_features+1,)
        Learned model weights.

    coef_ : ndarray of shape (n_features,)
            Learned model coefficients

    intercept_ : Scaler number
                 Learned model bias
    Notes
    -----
    - Uses gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs.
    - Intended for educational use; not as efficient as sklearn's LogisticRegression.
    """
    def __init__(self,learning_rate=.1,epochs = 1000):
        self.lr  = learning_rate
        self.epochs = epochs
        self.weights = None
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self,X):
        """
                            1
        Sigmoid(x) = ---------------
                       1 + exp(-x)

        """
        return 1/(1 + np.exp(-X))
    
    def fit(self,X,y):
        """
        Fit logistic regression model using gradient descent.

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
        X = np.insert(X, 0, 1, axis = 1)
        self.weights = np.zeros(X.shape[1])
        for i in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X,self.weights))
        
            # Calculating derivative wrt weights
            dw = (1/X.shape[0]) * np.dot((y_pred - y), X)

            ## Update weights
            self.weights -= self.lr*dw
        self.coef_ = self.weights[1:]
        self.intercept_ = self.weights[0]
        return self
    
    def predict(self,X):
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
        X = np.insert(X, 0, 1, axis = 1)
        preds =  self.sigmoid(np.dot(X,self.weights))
        return np.where(preds > 0.5, 1,0)