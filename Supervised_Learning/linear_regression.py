import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Helper Function: Plot Regression Line
def plot_regression_line(X, y, model, title):
    plt.scatter(X, y, color="blue", label="Actual")
    plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show(block=True)

# === Test 1: Synthetic Data === #
def synthetic_test():
    print("Running Synthetic Test...")

    # Synthetic Dataset
    X_synthetic = np.array([[1], [2], [3], [4], [5]])
    y_synthetic = np.array([2.1, 2.9, 4.2, 4.8, 6.1])

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_synthetic, y_synthetic)

    # Predictions
    predictions = model.predict(X_synthetic)
    print("Synthetic Data Predictions:", predictions)

    # Plot Regression Line
    plot_regression_line(X_synthetic, y_synthetic, model, 
                         "Linear Regression (Synthetic Data)")

# === Test 2: Breast Cancer Data === #
def real_data_test_binary():
    print("\nRunning Real Data Test (Binary Target)...")

    # Load Breast Cancer Dataset
    data = load_breast_cancer()
    X = data.data[:, :1]  # Use the first feature for simplicity
    y = data.target       # Binary target: 0 (malignant), 1 (benign)

    # Standardize the Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    print("Real Data Predictions (First 20):", predictions[:20])

    # Plot Regression Line
    plot_regression_line(X_train, y_train, model, 
                         "Linear Regression (Breast Cancer Data - Training Set)")

# === Test 3: California Housing Data === #
def real_data_test_continuous():
    print("\nRunning Real Data Test (Continuous Target)...")

    # Load California Housing Dataset
    data = fetch_california_housing()
    X = data.data[:, :1]  # Use the first feature for simplicity
    y = data.target       # Continuous target

    # Standardize the Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    print("Real Data Predictions (First 20):", predictions[:20])

    # Plot Regression Line
    plot_regression_line(X_train, y_train, model, 
                         "Linear Regression (California Housing Data - Training Set)")

# Run All Tests
if __name__ == "__main__":
    synthetic_test()
    real_data_test_binary()
    real_data_test_continuous()
