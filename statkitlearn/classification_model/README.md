# Classification Model

This section includes the main algorithms for **Classification**.

## Included Models

### **_LogisticRegression_**

**LogisticRegression** implements gradient descent to minimize the log loss error.

**Objective** - The prediction model is

$$
\hat{y} = \sigma(Xw)
$$

Where:-

- $\sigma$ is defined as:
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$
- $w$ is the weights matrix

**_Loss Function_**- the original idea is to maximise the maximum likelihood to find the optimum solution

$$
L(w) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

- But as the probabilities are small when we multiply them the production become insignificant. so instead we minimise something known as Log Loss error.

**_Log Loss Error_** is defined as

$$
L(w) = -\frac{1}{m}\sum_{i=1}^{m}(y_{\text{i}}log(\hat{y_{\text{i}}}) + (1 - y_{\text{i}})log(1-\hat{y_{\text{i}}}))
$$

**_Gradient_**:-

$$
\nabla_{w}L(w) = -\frac{1}{m}(y - \hat{y})X
$$

**_Weights Update Rule_**:- In each iteration

$$
w := w - \eta\nabla_{w}L(w)
$$

i.e.

$$
w := w + \frac{\eta}{m}(y - \hat{y})X
$$

where:-

- $\eta$ is learning rate

### User Guide

```python
from statkitlearn.classification_model import LogisticRegression ## Import model class
import numpy as np ## import numpy
from sklearn.datasets import make_classification ## Only to make sample data
from sklearn.model_selection import train_test_split ## To split training and test data

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30) ## make sample data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) ##Split train and test data

log_reg = LogisticRegression() ## Make object
log_reg.fit(X_train,y_train) ## fit training data
y_pred = log_reg.predict(X_test) ## Get Predictions for test data
print(log_reg.weights) ##Get the weights
```

### **_PerceptronClassifier_**

**PerceptronClassifier** implements step function to perform binary classification.

**_Algorithm_**:-

- The model predicts output using

$$
\hat{y} = step(w^Tx + b)
$$

- where step function is defined as

$$
\text{step}(z) =
\begin{cases}
1 & \text{if } z \ge 0 \\\\\
0 & \text{otherwise}
\end{cases}
$$

- **_Weight Update Rule_**:- for each missclassified sample $(x_{\text{i}},y_{\text{i}}):$

$$
w := w +\eta(y_{\text{i}} - \hat{y_{\text{i}}})x_{\text{i}}
$$

Where:-

- $w$ is the matrix of weights
- $\eta$ is learning rate
- $y_{\text{i}}$ is actual label
- $\hat{y_{\text{i}}}$ is predicted label

### User Guide

```python
from statkitlearn.classification_model import PerceptronClassifier ## Import model class
import numpy as np ## import numpy
from sklearn.datasets import make_classification ## Only to make sample data
from sklearn.model_selection import train_test_split ## To split training and test data

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30) ## make sample data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) ##Split train and test data

perc = PerceptronClassifier() ## Make object
perc.fit(X_train,y_train) ## fit training data
y_pred = perc.predict(X_test) ## Get Predictions for test data
print(perc.weights) ##Get the weights
```

### **_PerceptronClassifierSigmoid_**

**PerceptronClassifier** implements sigmoid function to perform binary classification.

**_Algorithm_**:-

- The model predicts output using

$$
\hat{y} = \sigma(w^Tx + b)
$$

- where sigmoid function is defined as

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**_Weight Update Rule_**:- for each missclassified sample $(x_{\text{i}},y_{\text{i}}):$

$$
w := w +\eta(y_{\text{i}} - \hat{y_{\text{i}}})x_{\text{i}}
$$

Where:-

- $w$ is the matrix of weights
- $\eta$ is learning rate
- $y_{\text{i}}$ is actual label
- $\hat{y_{\text{i}}}$ is predicted label

**_Why is it better then PerceptronClassifier_**:-

- A perceptron with a sigmoid activation is better because it produces smooth, differentiable outputs, allowing the use of gradient-based optimization for more stable and efficient learning.
- In contrast, the step function is non-differentiable and provides only hard binary outputs, making learning less flexible and limiting the model to only linearly separable problems.

### User Guide

```python
from statkitlearn.classification_model import PerceptronClassifierSigmoid ## Import model class
import numpy as np ## import numpy
from sklearn.datasets import make_classification ## Only to make sample data
from sklearn.model_selection import train_test_split ## To split training and test data

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30) ## make sample data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) ##Split train and test data

perc = PerceptronClassifier() ## Make object
perc.fit(X_train,y_train) ## fit training data
y_pred = perc.predict(X_test) ## Get Predictions for test data
print(perc.weights) ##Get the weights
```
