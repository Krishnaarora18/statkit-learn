# Linear Model

This section includes the main algorithms for **linear regression**.

## Included Models

### **_GDRegressor_**

**GDRegressor** implements Batch Gradient Descent to solve linear regression <br>

**Objective** - The Prediction model is

$$
\hat{y} = Xw
$$

**_Loss Functions_**: **GDRegressor** class currently have 3 different loss functions.

- **Mean Squared Error(MSE)**:-

$$
L = \frac{1}{2m}\sum_{i=1}^{m}(y_{\text{i}} - \hat{y_{\text{i}}})^2
$$

**Advantages** -

- Easiest to compute, smooth gradient
- Convex function
- Highly sensitive to small deviations, so model fits tightly

**Disadvantages**

- Extremely sensitive to outliers
- Not robust in real-world data

**Gradient**-

$$
\nabla_{\hat{y}}L = -\frac{1}{m}\sum_{i=1}^{m}(y_{i}- \hat{y_{i}})
$$

in matrix form:-

$$
\nabla_{\hat{y}}L = \frac{1}{m}(y - \hat{y})
$$

- **Pseudo Huber Loss**

$$
L = \frac{1}{m}\sum_{i=1}^{m}(\delta^2(\sqrt{1 + (\frac{y_{i}-\hat{y_{i}}}{\delta})^2} -1))
$$

**Advantages**:-

- Smooth version of Huber loss (unlike Huber, no kink)
- Behaves like MSE for small errors -> fast convergence
- Behaves like MAE for large errors -> robust to outliers
- Gradient does not explode like MSE

**Disadvantages**:-

- Requires choosing delta
- Slightly more expensive computationally

**Gradient**:-

$$
Let, error = y - \hat{y}
$$

$$
\therefore \nabla_{\hat{y}}L = \frac{error}{\sqrt{1 + (\frac{error}{\delta})^2}}
$$

- **Log Cosh Loss**

$$
L(w) = \frac{1}{m}\sum_{i=1}^{m}log(cosh(y_{i} - \hat{y_{i}}))
$$

$$
  where, cosh(x) = \frac{e^x + e^{-x}}{2}
$$

**Advantages**

- Very smooth loss (smoother than pseudo-Huber)
- Robust to outliers like MAE.
- Behaves like MSE near zero.
- No hyperparameter (unlike Pseudo-Huber’s $\delta$)

**Disadvantages**

- Slightly slower than MSE
- Less interpretable (not commonly taught)

**Gradient**

$$
\nabla_{\hat{y}}L = tanh(y - \hat{y})
$$

$$
where, tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Regularisation**:-
To overcome the problem of overfitting the **GDRegressor** have 3 regularisation techniques

- **L1 Regularisation**- In this we add a penalty term $\alpha||w||$ due to which the coefficients of unwanted columns shrinks to 0

$$
L_{new} = L_{old} + \alpha||w||
$$

- **L2 Regularisation**- we add a penalty term $\alpha||w||^2$. due to which the coefficients of unwanted columns start shrinking but unlike L1 regularisation the coefficients does not shrink to 0

$$
L_{new} = L_{old} + \alpha||w||^2
$$

- **ElasticNet Reguralisation**- It is a mix of both L1 and L2 regularisation, it is less robust then L1 regularisation and more robust then L2 Regularisation.

$$
L_{new} = L_{old} +\lambda(\alpha||w||_1 + (1-\alpha)||w||_2^2)
$$

**_Algorithm_**

- Initialises the weights to be 0 then,
- Compute the predictions

$$
\hat{y} = Xw + b
$$

- Compute the error

$$
error = y - \hat{y}
$$

- Compute the Gradient of the choosen Loss Function wrt $\hat{y}$

$$
calculate, \nabla_{\hat{y}}L
$$

- Compute Gradient wrt $w$ by chain rule

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}}\cdot\frac{\partial \hat{y}}{\partial w}
$$

i.e.

$$
\frac{\partial L}{\partial w} = -X^T\cdot\frac{\partial L}{\partial \hat{y}}
$$

- Compute gradient wrt to $b$

$$
\frac{\partial L}{\partial b} = - \frac{1}{m}\sum_{i=1}^{m}(\frac{\partial L}{\partial \hat{y_i}})
$$

- Add the gradient of penalty term if choosen

$$

\frac{\partial L}{\partial w} = \frac{\partial L}{\partial w}+\nabla_{w}penalty


$$

- Calculate and save loss
- Update weights

$$

w := w - \eta\cdot\frac{\partial L}{\partial w}


$$

- Update bias

$$
b := b - \eta \cdot \frac{\partial L}{\partial b}
$$

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

\theta\_{\text{ridge}} = (X^{T}X + \alpha I)^{-1} X^{T}y


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
gdr = GDRegressor(loss="log_cosh",learning_rate = 0.2) ##Make an object
gdr.fit(X_train,y_train) ##fit the training data
print(gdr.coef_,gdr.intercept_) ##Print the coefficients and intercept
y_pred = gdr.predict(X_test) ## Get the predictions for test data
```

---

### 2). Stochastic Gradient Descent Regressor

```python
from statkitlearn.linear_model import SGDRegressor #Import the SGDRegressor Class
sgdr = SGDRegressor() ##Make an object
sgdr.fit(X_train,y_train) ##fit the training data
print(sgdr.weights,sgdr.bias) ##Print the coefficients and intercept
y_pred = sgdr.predict(X_test) ## Get the predictions for test data
```

---

### 3). Ridge Regression

```python
from statkitlearn.linear_model import RidgeRegressor #Import the RidgeRegressor Class
rr = RidgeRegressor() ##Make an object
rr.fit(X_train,y_train) ##fit the training data
print(rr.weights,rr.bias) ##Print the coefficients and intercept
print(rr.weights,rr.bias) ##Print the weights and bias
y_pred = rr.predict(X_test) ## Get the predictions for test data
```

$$
$$
