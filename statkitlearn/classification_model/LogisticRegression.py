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

    probability_threshold : float default = 0.5
        the pobabilty after which model will predict 1.

    Attributes
    ----------
    weights : ndarray of shape (n_features+1,)
        Learned model weights.

    bias : Scaler number
        Learned model bias
    Notes
    -----
    - Uses gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs.
    - Intended for educational use; not as efficient as sklearn's LogisticRegression.
    """
    def __init__(self,learning_rate=.1,epochs = 1000, probablity_threshold = 0.5):
        self.lr  = learning_rate
        self.epochs = epochs
        self.weights = None

    def sigmoid(self,X):
        """
                            1
        Sigmoid(x) = ---------------
                       1 + exp(-x)

        """
        return 1/(1 + np.exp(-X))
    
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
        
        if not isinstance(self.lr, (int,float)) or self.lr <= 0:
            raise ValueError("Learning rate must be a positive real number")
        
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Epochs must be a positive integer")
        
        if not np.all(np.isfinite(X_train)):
            raise ValueError("X_train contains NaN or infinite values")

        if not np.all(np.isfinite(y_train)):
            raise ValueError("y_train contains NaN or infinite values")
        
        if not np.issubdtype(y_train.dtype, np.number):
            raise ValueError("y_train must be numeric for regression")
    
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
        X,y = self.validate_fit(X,y)
        self.bias = 0
        self.weights = np.zeros(X.shape[1])
        for i in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X,self.weights) + self.bias)
        
            # Calculating derivative wrt weights
            dw = (1/X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1/X.shape[0]) * np.sum(y_pred - y)

            ## Update weights and bias
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
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
        preds =  self.sigmoid(np.dot(X,self.weights) + self.bias)
        return np.where(preds > 0.5, 1,0)