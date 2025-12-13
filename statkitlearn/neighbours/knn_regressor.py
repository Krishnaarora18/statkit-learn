import numpy as np

class KNNRegressor:
    """
    K-Nearest Neighbors (KNN) regressor implemented from scratch using NumPy.

    This regressor predicts a continuous target value for a test sample
    by computing a weighted average of the k nearest neighbors' target values.
    The weights are computed using an exponential decay based on distance.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to consider.

    distance_type : str, default="euclidean"
        Distance metric to use. Supported values are:
        - "euclidean"
        - "manhattan"
        - "minowski"
        - "cosine"

    p : int or float, default=0
        Power parameter for Minkowski distance.
        Must be greater than 0 when using Minkowski distance.
    """

    def __init__(self, k=5, distance_type="euclidean", p=0):
        """
        Initialize the KNN regressor with hyperparameters.
        """
        self.k = k
        self.train_points = None
        self.train_labels = None
        self.p = p
        self.distance_type = distance_type

    def distance(self, X_train, X_test):
        """
        Compute the distance between a training point and a test point.

        Parameters
        ----------
        X_train : numpy.ndarray
            A single training data point.

        X_test : numpy.ndarray
            A single test data point.

        Returns
        -------
        float
            The computed distance based on the selected distance metric.

        Raises
        ------
        ValueError
            If an unsupported distance metric is provided or
            if p <= 0 for Minkowski distance.
        """
        if self.distance_type == "euclidean":
            return np.sqrt(np.sum((X_train - X_test) ** 2))
        elif self.distance_type == "manhattan":
            return np.sum(np.abs(X_train - X_test))
        elif self.distance_type == "minowski":
            if self.p <= 0:
                raise ValueError("Value of p must be greater then 0")
            return np.pow(np.sum(np.abs(X_train - X_test) ** self.p), (1 / self.p))
        elif self.distance_type == "cosine":
            return (1 - np.dot(X_train, X_test) /
                    (np.sqrt(np.sum(X_train ** 2)) * np.sqrt(np.sum(X_test ** 2))))
        else:
            raise ValueError("Unknown distance metric")

    def fit(self, X_train, y_train):
        """
        Store the training data and target values.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature matrix of shape (n_samples, n_features).

        y_train : numpy.ndarray
            Target values of shape (n_samples,).
        """
        self.train_points = X_train
        self.train_labels = y_train

    def predict(self, X_test):
        """
        Predict continuous target values for test data.

        For each test sample, the k nearest neighbors are selected and
        a weighted mean of their target values is computed using
        exponential distance-based weights.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test feature matrix of shape (n_test_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted continuous values for each test sample.
        """
        preds = []
        for i in range(X_test.shape[0]):
            distances = []
            for j in range(self.train_points.shape[0]):
                distance = self.distance(self.train_points[j], X_test[i])
                distances.append([distance, self.train_labels[j]])

            sorted_distances = sorted(distance, key=lambda x: x[0])[:self.k]
            values = np.array([value[1] for value in sorted_distances])
            dist = np.array([distance[0] for distance in sorted_distances])
            weights = np.exp(-dist)
            preds.append(np.sum(weights * values) / np.sum(weights))
        return np.array(preds)