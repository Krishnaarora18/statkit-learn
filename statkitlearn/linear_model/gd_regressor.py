import numpy as np

class GDRegressor:
    """
    Linear Regression using Full-Batch Gradient Descent.

    This estimator minimizes the selected loss using gradient descent
    updates on the entire training dataset per iteration.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each gradient descent update.

    epochs : int, default=1000
        Number of full gradient descent iterations.
    
    loss : str, to select which loss function to minimize
        Currently included Loss functions
            - mse (Mean Squared Error)
            - pseudo_huber (Peusdo Huber Loss)
            - log_cosh (Log Cosh Loss)
        default="mse"  

    delta : float, default=1.5
        Used to compute Pseudo Huber Loss

    penalty : str, default = None
        To select whether to apply penalty to the weights
        to reduce risk of overfitting
        Currently included penalties
           - l1 (Lasso Penalty)
           - l2 (Ridge Penalty)
           - elasticnet (ElasticNet Penalty(Combination of both L1 and L2 penalty))

    alpha : float, default=0.1
        Regularization strenght of L1 and L2 penalty

    l1_ratio : float, default=0.5
        Used to compute elasticnet penalty

    Attributes
    ----------
    weights : ndarray of shape (n_features+1,)
        Learned model weights.
    intercept_ : Scaler number
        Bias of the model.
    coef_ : ndarray of shape (n_features,)
        Learned model coefficients.
    loss_history_ : list of float
        Stores the MSE loss value for each epoch.

    Notes
    -----
    - Uses full-batch gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs and loss function.
    - Normalising the dataset is strongly recommended to obtain optimum results.
    - Intended for educational use; not as efficient as closed-form.
    """
    
    def __init__(self, learning_rate=0.1, epochs=1000,loss = "mse",delta = 1.5,penalty = None,alpha = 0.1,l1_ratio=0.5):
        self.epochs = epochs
        self.lr = learning_rate
        self.penalty = penalty
        self.weights = None
        self.delta = delta
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None
        self.loss = loss
        self.loss_history_ = [] 

    def gradient(self, error):
        """
        Computes the gradient of selected loss function wrt y_pred 
        """
        if self.loss == "mse":
            return error
        elif self.loss == "pseudo_huber":
            return error/np.sqrt(1 + (error/self.delta)**2)
        elif self.loss == "log_cosh":
            return np.tanh(error)
        else:
            raise ValueError("Unknown Loss Function")
        
    def mse(self,error):
        """
        Computes the mean squared error 
        MSE = summation(error^2)/Total Samples
        """
        return np.mean((error)**2)
    
    def pseudo_huber(self,error):
        """
        Computes the pseudo huber loss
        PHL =  δ^2(sqrt(1 + (error/ δ)^2) - 1)
        """
        return np.mean(self.delta**2 * (np.sqrt(1 + (error/self.delta)**2) - 1))
    
    def log_cosh(self, error):
        """
        Computes the Log Cosh Loss
        LChL = summation(log(cosh(error)))/Total Samples
        """
        return np.mean(np.log(np.cosh(error)))
    
    def fit(self, X_train, y_train):
        """
        Fit Gradient descent model to calculate weights which minimize the selected loss.

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
        
        X_train = np.insert(X_train, 0, 1, axis=1)
        self.weights = np.zeros(X_train.shape[1])
        

        for i in range(self.epochs):
            y_pred = np.dot(X_train, self.weights) ## Calculate predicted value of y
            error = y_train - y_pred ## Compute error

            dy_pred = self.gradient(error) ## Compute Gradient of Loss function wrt y_pred
            dw = - np.dot(X_train.T, dy_pred)/X_train.shape[0] ## Compute Gradient of Loss function wrt weights

            # Skip bias term
            w = self.weights.copy()
            w[0] = 0

            ## Apply Penalty
            if self.penalty == None:
                dw = dw 
            elif self.penalty == "l2":
                dw += (2*self.alpha/X_train.shape[0])*w
            elif self.penalty == "l1":
                dw += (self.alpha/X_train.shape[0]) * np.sign(w)
            elif self.penalty == "elasticnet":
                dw += (self.alpha/X_train.shape[0]) * (self.l1_ratio * np.sign(w) + 2 * (1 - self.l1_ratio) * w)
            else:
                raise ValueError("Invalid Penalty")

            ## Calculate Loss
            if self.loss == "mse":
                loss = self.mse(error)  
            elif self.loss == "pseudo_huber":
                loss = self.pseudo_huber(error)
            elif self.loss == "log_cosh":
                loss = self.log_cosh(error)
            else:
                raise ValueError("Unknown Loss Function")

            self.loss_history_.append(loss) ## Save the loss per epoch

            self.weights -= self.lr * dw ## Update weights


        self.intercept_ = self.weights[0] ## get intercept(or bias)
        self.coef_ = self.weights[1:] ## Get coefficients
        print(self.intercept_,self.coef_)

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
        X_test = np.insert(X_test, 0, 1, axis=1) ## insert a column of 1 at position 0
        return np.dot(X_test, self.weights) ## Return the predictions
