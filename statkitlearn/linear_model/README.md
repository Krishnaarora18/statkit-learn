# Linear Model

This section includes the main algorithms for **linear regression**.

## Included Models

### **_GDRegressor_**

**GDRegressor** implements Batch Gradient Descent to solve linear regression <br><br>
**Loss Function(mean squared error)** - Gradient Descent minimizes the mean squared error:

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} (X^{(i)}\theta - y^{(i)})^2
$$

**Gradient of the loss function**-The gradient of the loss with respect to parameters θ is:

$$
\nabla_{\theta} J(\theta) = \frac{1}{m}X^{T}(X\theta - y)
$$

**Gradient Descent Update Rule**-Parameters are updated iteratively using:

$$
\theta := \theta - \eta \cdot \nabla_{\theta}J(\theta)
$$

Substituting the gradient expression:

$$
\theta := \theta - \eta \cdot \frac{1}{m}X^{T}(X\theta - y)
$$

where:-

- **η** is the learning rate
- **m** is number of samples

**Hyper-Parameters**:-

- **_learning_rate_** - default = 0.1, controls the rate at which model updates
- **_epochs_** - default = 100, number of iterations

---

### **_SGDRegressor_**

**SGDRegressor** implements Stochastic Gradient Descent to solve linear regression<br>
Instead of using all samples to compute the gradient (like GD), it uses one sample at a time.

**Objective** - The Prediction model is

$$
\hat{y} = X\theta
$$

**Loss Function(MSE)** - Same mean squared error loss

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} (X^{(i)}\theta - y^{(i)})^2
$$

**Stochastic Gradient** -
For a single training sample ( 𝑥(𝑖), 𝑦(𝑖) ), the gradient of the loss is:

$$
g^{(i)} = x^{(i)T}(x^{(i)}\theta - y^{(i)})
$$

**SGD Update Rule** - Each iteration updates parameters using only one data point:

$$
\theta := \theta - \eta \cdot g^{(i)}
$$

$$
\theta := \theta - \eta \cdot x^{(i)T}(x^{(i)}\theta - y^{(i)})
$$

**Hyper-Parameters**:-

- **_learning_rate_** - default = 0.1, controls the rate at which model updates.
- **_epochs_** - default = 100, number of iterations.

---

### **_RidgeRegressor_**

**RidgeRegressor** implements L2-regularized linear regression, which prevents overfitting by penalizing large weights.
**Objective** - The Prediction model is

$$
\hat{y} = X\theta
$$

**Ridge Loss Function** - Ridge adds an **L2 penalty** to the MSE loss:

$$
J(\theta) = \frac{1}{2m}(y - X\theta)^T(y - X\theta) + \frac{\alpha}{2m}\theta^T\theta
$$

where

- α = regularization strength (lambda).

**Closed Form Solution** - Ridge regression has a closed-form solution:

$$
\theta_{\text{ridge}} = (X^{T}X + \alpha I)^{-1} X^{T}y
$$

where

- I is the identity matrix.

**Effect of Regularization** <br>
The penalty term

$$
\frac{\alpha}{2m}\theta^T\theta
$$

- Reduces Coefficients.
- Reduces Variance.

## How to Use?

### 1). Linear Regression(GDRegressor)

```python
from statkitlearn.linear_model import GDRegressor #Import the GDRegressor Class
gdr = GDRegressor() ##Make an object
gdr.fit(X_train,y_train) ##fit the training data
print(gdr.weights,gdr.bias) ##Print the weights and bias
y_pred = gdr.predict(X_test) ## Get the predictions for test data
```

---

### 2). Stochastic Gradient Descent Regressor

```python
from statkitlearn.linear_model import SGDRegressor #Import the SGDRegressor Class
sgdr = SGDRegressor() ##Make an object
sgdr.fit(X_train,y_train) ##fit the training data
print(sgdr.weights,sgdr.bias) ##Print the weights and bias 
y_pred = sgdr.predict(X_test) ## Get the predictions for test data
```

---

### 3). Ridge Regression

```python
from statkitlearn.linear_model import RidgeRegressor #Import the RidgeRegressor Class
rr = RidgeRegressor() ##Make an object
rr.fit(X_train,y_train) ##fit the training data
print(rr.weights,rr.bias) ##Print the weights and bias
y_pred = rr.predict(X_test) ## Get the predictions for test data
```
