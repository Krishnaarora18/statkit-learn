import numpy as np
from statkitlearn.preprocessing import OutliersRemover

X = np.array([
    [10, 200],
    [12, 220],
    [11, 215],
    [500, 300],   # Outlier
])

rem_out = OutliersRemover(factor=1.5)
clean = rem_out.fit_transform(X)

print(clean)
