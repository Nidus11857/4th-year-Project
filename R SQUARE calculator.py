import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Simulate data for two cases:
# Case 1: Wide range of x
np.random.seed(0)
x_wide = np.linspace(0, 100, 1000)
y_wide = 5 * x_wide + np.random.normal(0, 20, size=len(x_wide))  # y = 5x + noise

# Case 2: Narrow range of x
x_narrow = np.linspace(48, 52, 1000)
y_narrow = 5 * x_narrow + np.random.normal(0, 20, size=len(x_narrow))  # same noise level

# Fit models
model_wide = LinearRegression().fit(x_wide.reshape(-1, 1), y_wide)
model_narrow = LinearRegression().fit(x_narrow.reshape(-1, 1), y_narrow)

# Calculate R^2
r2_wide = r2_score(y_wide, model_wide.predict(x_wide.reshape(-1, 1)))
r2_narrow = r2_score(y_narrow, model_narrow.predict(x_narrow.reshape(-1, 1)))

r2_wide, r2_narrow
