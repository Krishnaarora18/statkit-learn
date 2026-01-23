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

    optimizer : str, to select optimizer
        Currently included Optimizers
            - adamax (AdaMax)
            - Adam (Adam)
            - rmsprop (RMSProp)
            - momentum (Momentum)
        default = None (No optimizer is used)

    momentum : float, to adjust the value of beta in optimizers
        default = 0.9

    scaling_decay = float, used to adjust the value of scaling decay in adam optimizer
        default = 0.999
                    
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
    bias : Scaler number
        Learned model bias
    loss_history_ : list of float
        Stores the MSE loss value for each epoch.

    Notes
    -----
    - Uses full-batch gradient descent; gradient is computed on all samples.
    - Convergence depends strongly on learning rate and number of epochs and loss function.
    - Normalising the dataset is strongly recommended to obtain optimum results.
    - Intended for educational use; not as efficient as closed-form.
    """
    
    def __init__(self, learning_rate=0.1, epochs=1000,
                 loss = "mse",delta = 1.5,penalty = None,
                 alpha = 0.1,l1_ratio=0.5, momentum = 0.9,
                 epsilon = 1e-8, optimizer = None, scaling_decay = 0.999):
        self.epochs = epochs
        self.lr = learning_rate
        self.penalty = penalty
        self.weights = None
        self.delta = delta
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.bias = None
        self.loss = loss
        self.mw = 0
        self.mb = 0
        self.sw = 0
        self.sb = 0
        self.e = epsilon
        self.beta = momentum
        self.beta2 = scaling_decay
        self.mw_hat = 0
        self.mb_hat = 0
        self.sw_hat = 0 
        self.sb_hat = 0
        self.loss_history_ = [] 
        self.tol = 1e-4 
        self.n_iter_no_change = 10 
        self.optimizer = optimizer


        if self.penalty not in {None, "l1", "l2", "elasticnet"}:
            raise ValueError("Unknown Penalty")
        if self.loss not in {"mse","log_cosh","pseudo_huber"}:
            raise ValueError("Unknown Loss Function")
        if self.optimizer not in ["rmsprop","adagrad","adam","momentum"]:
            raise ValueError("Unknown Optimizer")

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
        
        if not isinstance(self.alpha, (int,float)) or self.alpha < 0:
                raise ValueError("alpha must be 0 or a positive real number")
        
        if not isinstance(self.l1_ratio, (int,float)) or not (0 <= self.l1_ratio <= 1):
                raise ValueError("l1_ratio must be 0 or a positive real number")

        return X_train, y_train

    def gradient(self, error):
        """
        Computes the gradient of selected loss function wrt y_pred 
        """
        if self.loss == "mse":
            return error
        elif self.loss == "pseudo_huber":
            if not isinstance(self.delta, (int,float)) or self.delta <= 0:
                raise ValueError("delta must be a positive real number")
            return error/np.sqrt(1 + (error/self.delta)**2)
        elif self.loss == "log_cosh":
            return np.tanh(error)
        
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
        if not isinstance(self.delta, (int,float)) or self.delta <= 0:
            raise ValueError("delta must be a positive real number")
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
        
        X_train, y_train = self.validate_fit(X_train, y_train)

        self.bias = 0
        self.weights = np.zeros(X_train.shape[1])

        self.mw = self.mb = self.sw = self.sb = 0

        
        for i in range(self.epochs):
            y_pred = np.dot(X_train, self.weights) + self.bias ## Calculate predicted value of y
            error = y_train - y_pred ## Compute error

            dy_pred = self.gradient(error) ## Compute Gradient of Loss function wrt y_pred
            dw = - np.dot(X_train.T, dy_pred)/X_train.shape[0] ## Compute Gradient of Loss function wrt weights
            db = - np.mean(dy_pred)

            w = self.weights.copy()


            ## Apply Penalty
            if self.penalty == None:
                dw = dw 
            elif self.penalty == "l2":
                dw += (self.alpha/X_train.shape[0])*w
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

            self.loss_history_.append(loss) ## Save the loss per epoch

            if i > self.n_iter_no_change:
                recent_losses = self.loss_history_[-self.n_iter_no_change:]
                loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
                if abs(loss_improvement) < self.tol:
                    print(f"Converged at epoch {i}")
                    break

            ## Update weights 
            if self.optimizer == None:
                self.weights -= self.lr*dw
                self.bias -= self.lr*db

            elif self.optimizer == "momentum":
                self.mw = self.beta*self.mw - self.lr*dw
                self.mb = self.beta*self.mb - self.lr*db

                self.weights += self.mw
                self.bias += self.mb

            elif self.optimizer == "adagrad":
                self.sw += dw**2 
                self.sb += db**2
                
                self.weights -= self.lr * dw/(np.sqrt(self.sw + self.e))
                self.bias -= self.lr * db/(np.sqrt(self.sb + self.e))

            elif self.optimizer == "rmsprop":
                self.sw  = self.beta*self.sw + (1 - self.beta) * dw ** 2
                self.sb = self.beta * self.sb + (1 - self.beta) * db ** 2

                self.weights -= self.lr * dw/(np.sqrt(self.sw + self.e))
                self.bias -= self.lr * db/(np.sqrt(self.sb + self.e))

            elif self.optimizer == "adam":
                self.mw = self.beta*self.mw + (1 - self.beta) * dw
                self.mb = self.beta*self.mb + (1 - self.beta) * db

                self.sw = self.beta2*self.sw + (1 - self.beta2) * dw ** 2
                self.sb = self.beta2*self.sb + (1 - self.beta2) * db ** 2

                t = i + 1

                self.mw_hat = self.mw/(1 - self.beta ** t)
                self.mb_hat = self.mb/(1 - self.beta ** t)

                self.sw_hat = self.sw/(1 - self.beta2 ** t)
                self.sb_hat = self.sb/(1 - self.beta2 ** t)

                self.weights -= self.lr * self.mw_hat/np.sqrt(self.sw_hat + self.e)
                self.bias -= self.lr * self.mb_hat/np.sqrt(self.sb_hat + self.e)

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
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_test = self.validate_X(X_test)
        if X_test.shape[1] != len(self.weights):
            raise ValueError(f"X_test has {X_test.shape[1]} features, but model was trained with {len(self.weights)} features")

        return np.dot(X_test, self.weights) + self.bias ## Return the predictions
