import numpy as np

class KNNClassifier:
    """
    This classifier predicts the class of a test sample based on the majority
    class among its k nearest neighbors in the training dataset.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to consider for classification.

    distance_type : str, default="euclidean"
        Distance metric to use. Supported values are:
        - "euclidean"
        - "manhattan"
        - "minowski"
        - "cosine"

    p : int or float, default=1
        Power parameter for Minkowski distance.
        Must be greater than 0 when using Minkowski distance.
    """

    def __init__(self, k=5, distance_type="euclidean", p=1):
        """
        Initialize the KNN classifier with hyperparameters.
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
            raise ValueError("Unknown Distance metric")

    def fit(self, X_train, y_train):
        """
        Store the training data and labels.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature matrix of shape (n_samples, n_features).

        y_train : numpy.ndarray
            Training labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If the number of samples in X_train and y_train do not match.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must be equal")
        self.train_points = X_train
        self.train_labels = y_train

    def predict(self, X_test):
        """
        Predict class labels for test data.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test feature matrix of shape (n_test_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted class labels for each test sample.
        """
        preds = []
        for i in range(X_test.shape[0]):
            distance_list = []
            for j in range(self.train_points.shape[0]):
                distance = self.distance(self.train_points[j], X_test[i])
                distance_list.append([distance, self.train_labels[j]])
            distance_list = sorted(distance_list)[:self.k]
            categories = [category[1] for category in distance_list]
            preds.append(np.bincount(categories).argmax())

        return np.array(preds)