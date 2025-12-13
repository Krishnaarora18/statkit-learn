# Neighbours

This section cover algorithms which predicts the output feature by seeing its neighbours.

## Included Models

### **_KNNClassifier_**

This classifier predicts the class of a test sample based on the majority class among its k nearest neighbors in the training dataset.

**_Distance_**

There are a number of types of distances which we can use to find the nearest neighbours:

- Euclidean Distance
  - Measures straight-line distance in space
  - Most common in KNN
  - Sensitive to scale of features (may need normalization)

$$
d(x,y) = \sqrt{\sum_{i=1}^n(x_i - y_i)^2}
$$

- Manhattan Distance
  - Sum of absolute differences
  - Moves along grid lines
  - Can be better for “per-feature” differences.

$$
d(x,y) = \sum_{i=1}^{n}|x_i - y_i|
$$

- Minowski Distance
  - Generalization of Euclidean (p=2) and Manhattan (p=1)
  - You can choose p depending on problem

$$
d(x,y) = (\sum_{i=1}^{n}|x_i - y_i|^p)^{1/p}
$$

- Cosine distance
  - Measures angle between vectors, ignores magnitude
  - Often used in text / high-dimensional sparse data

$$
d(x,y) = 1 - \frac{x\cdot y}{|x||y|}
$$

**_Algorithm_**

- Fit the train data (nothing happens in fitting data as KNN is a lazy learning algorithm).
- Predict the test data.
- for prediction of every point the algorithm calculates distance of individual test point with all the train data points.
- Then the distances are sorted in ascending order and smallest k distances are takes.
- Then the algorithms finds the most common label in the given points (majority voting).

### **_KNNRegressor_**

- Same as KNNClassifier but as the model have to predict continous data after finding the points with least distance.
- it calculates there weights

  $$
  w_i = e^{-d_i}
  $$

- Where $d_i$ is the distance of the point with test data point
- after that we calculate weighted mean of the k points i.e. prediction

$$
\hat{y} = \frac{\sum_{i=1}^{k}w_i y_i}{\sum_{i=1}^{k}w_i}
$$

### User Guide

```python
from statkitlearn.neighbours import KNNClassifier
import numpy as np

classifier = KNNClassifier(distance_type = "manhattan", k=5)
classifier.fit(X_train, y_train) ##Fit your training data
classifier.predict(X_test) ##Get predictions on you test data
```
