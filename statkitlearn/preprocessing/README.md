# Preprocessing

This section includes classes useful for **_Data Preprocessing_**

## Currently Included

- StandardScaler
- OutliersRemover

## Standard Scaler

Standardizes numerical features by removing the mean and scaling to unit variance:

$$
X_{\text{scaled}} = \frac{X - \mu}{\sigma_{\text{X}}}
$$

### Features

- Fully **vectorized**
- stores **mean\_** and **stdev\_** for inverse transformation

### User Guide

```python
from statkitlearn.preprocessing import StandardScaler #import class
import numpy as np
X = np.array([[1, 2],
              [2, 3],
              [3, 4]]) ##Create sample data

scaler = StandardScaler() ## Create Object
scaler.fit(X) ##fit the data
X_scaled = scaler.transform(X) ##Transform the data
X_inverse_scaled = scaler.inverse_transform(X) ##Get the original data
```

## OutliersRemover

Remove outliers from the data by creating upper and lower fence and then remove datapoints outside fences

Q1 = 25th percentile<br>
Q3 = 75th percentile<br>
IQR = Q3 - Q1<br>
upper fence = Q3 + factor*IQR<br>
lower fence = Q1 - factor*IQR<br>
<br>

- Removes datapoints > Upper fence and < lower fence

### User Guide

```python
from statkitlearn.preprocessing import OutliersRemover
import numpy as np

X = np.array([
    [10, 200],
    [12, 220],
    [11, 215],
    [500, 300],   # Create sample data
])

rem_out = OutliersRemover(factor=1.5) ##create object
clean = rem_out.fit_transform(X) ##fit and transform data
```

---

- **_More features will be added_**
