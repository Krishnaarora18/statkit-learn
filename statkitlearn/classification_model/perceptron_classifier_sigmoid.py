import numpy as np

class PerceptronClassifierSigmoid:
    """
    ==========
    Perceptron
    ==========
    This is a simple perceptron model performing bi-class classification.
    This model uses step function to classify points
    =======
    Sigmoid(X)
    =======
              1
    X = -------------
         1 + exp(-x)

    ==========
    Attributes
    ==========
    Learning Rate: Decides the learning rate of Algorithm 
                   default: 0.01
            
    epochs: Decides how many time to repeat the process
            default: 1000
    """
    def __init__(self,learning_rate=0.01,epochs=1000):
        
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.coef_ = None
        self.intercept_ = None
    
    def sigmoid(self,X):
        return 1/(1 + np.exp(-X))
     

    def fit(self,X_train,y_train):
        """
        Fits the training data
        first inserts a column of values 1 at index 0 in training dataset
        to obtain the standard equation:
        sum(wixi) = 0
        then runs a loop in range of epochs 
        inside each loop a random index is selected 
        then a predicted point is calculated using current weights 
        by calculating the dot product of train data of random index and current weights
        and then sigmoid function is applied 

        after that the Weights are updated 
        """
        X_train = np.insert(X_train,0,1,axis=1)
        self.weights = np.zeros(X_train.shape[1])
        for i in range(self.epochs):
            idx = np.random.randint(X_train.shape[0])
            y_pred = self.sigmoid(np.dot(X_train[idx],self.weights)) 
            self.weights = self.weights + self.lr*(y_train[idx] - y_pred)*X_train[idx]
            

        self.coef_ = self.weights[1:]
        self.intercept_ = self.weights[0]
        return self
    
    def predict(self,X_test):
        """
        Fits the Test data and predicts the Y values by the sigmoid
        function of dot procduction of test data with weights
        """

        X_test = np.insert(X_test,0,1,axis=1)
        preds = np.dot(X_test, self.weights)
        
        probs = self.sigmoid(preds)
        return np.where(probs >= 0.5, 1, 0) 