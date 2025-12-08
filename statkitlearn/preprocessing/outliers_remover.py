import numpy as np

class OutliersRemover:
    """
    ----------------
    Outliers Remover
    ----------------
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Upper Fence = Q3 + factor*IQR
    Lower Fence = Q1 - factor*IQR

    --------
    Criteria
    --------
    Remove Data points which are greater then upper fence and lower then lower fence

    default factor = 1.5
    """
    def __init__(self,factor=1.5):
        self.Q1 = None
        self.Q3 = None
        self.factor = factor
        self.upper_fence = None
        self.lower_fence = None
        self.IQR = None
    def fit(self,X):
        """
        Fit the data to remove outliers
        """
        self.Q1 = np.percentile(X, 25, axis=0)
        self.Q3 = np.percentile(X, 75, axis=0)
        self.IQR = self.Q3 - self.Q1
        self.upper_fence = self.Q3 + self.factor*self.IQR
        self.lower_fence = self.Q1 - self.factor*self.IQR

        return self
    
    def transform(self, X):
        """
        Remove rows containing outliers in any feature.
        """
        cleaned = np.all((X >= self.lower_fence) & (X <= self.upper_fence), axis=1)
        return X[cleaned]
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)