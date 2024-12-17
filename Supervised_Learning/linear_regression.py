import matplotlib
matplotlib.use("Agg")  # Set the non-interactive backend (before importing pyplot)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.legend()
plt.title("Linear Regression Test")
plt.xlabel("X")
plt.ylabel("y")

# Save plot as a file (no display window needed)
plt.savefig("linear_regression_test.png")
print("Plot saved as 'linear_regression_test.png'")
