import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Helper Function: Plot Decision Boundary
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# === Test 1: Synthetic Data === #
def synthetic_test():
    print("Running Synthetic Test...")

    # Synthetic Dataset
    X_synthetic = np.array([[1, 2], [2, 3], [3, 4], [5, 5], [6, 5], [7, 4]])
    y_synthetic = np.array([0, 0, 0, 1, 1, 1])

    # KNN Classifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_synthetic, y_synthetic)

    # Predictions
    predictions = model.predict(X_synthetic)
    print("Synthetic Data Predictions:", predictions)

    # Plot Decision Boundary
    plot_decision_boundary(X_synthetic, y_synthetic, model, 
                           "KNN Decision Boundary (Synthetic Data)")

# === Test 2: Real Breast Cancer Data === #
def real_data_test():
    print("\nRunning Real Data Test...")

    # Load Breast Cancer Dataset
    data = load_breast_cancer()
    X = data.data[:, :2]  # Use the first two features for visualization
    y = data.target       # Binary labels: 0 (malignant), 1 (benign)

    # Standardize the Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN Classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    print("Real Data Test Predictions (First 20):", predictions[:20])

    # Plot Decision Boundary
    plot_decision_boundary(X, y, model, 
                           "KNN Decision Boundary (Breast Cancer Data)")

# Run Both Tests
if __name__ == "__main__":
    synthetic_test()
    real_data_test()
