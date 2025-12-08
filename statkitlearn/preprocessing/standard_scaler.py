import numpy as np

class StandardScaler:
    """
                (X - mean(X))
    Scaled(X) = -------------
                    std(X)
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """
        Compute the mean and std for each feature.
        X : numpy array (n_samples, n_features)
        """
        self.mean_ = np.mean(X, axis=0)  ## Calculate mean
        self.stdev_ = np.std(X, axis=0, ddof=0) ## Calculate Standard deviation
        
        # Avoiding division by zero
        self.stdev_[self.stdev_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """
        Apply standardization: (X - mean) / std
        """
        if self.mean_ is None or self.stdev_ is None:
            raise Exception("Fit the data first to perform transformation")
        
        return (X - self.mean_) / self.stdev_
    
    def fit_transform(self, X):
        """
        Fit to data, then return transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Convert standardized data back to original scale.
        """
        return X_scaled * self.stdev_ + self.mean_