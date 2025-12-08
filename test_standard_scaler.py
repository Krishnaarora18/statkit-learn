import numpy as np
from statkitlearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = np.array([[1, 2],
              [2, 3],
              [3, 4]])

X_scaled = scaler.fit_transform(X)
print("Scaled:\n", X_scaled)

X_original = scaler.inverse_transform(X_scaled)
print("Original:\n", X_original)
