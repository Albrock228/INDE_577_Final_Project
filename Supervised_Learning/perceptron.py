import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Perceptron Class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0
                if condition:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        return np.where(output <= 0, 0, 1)

# Helper function: Plot Decision Boundary
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
    X_synthetic = np.array([[1, 2], [2, 3], [3, 4], [5, 5], [6, 5], [7, 4]])
    y_synthetic = np.array([0, 0, 0, 1, 1, 1])

    perceptron = Perceptron(learning_rate=0.1, n_iterations=50)
    perceptron.fit(X_synthetic, y_synthetic)

    predictions = perceptron.predict(X_synthetic)
    print("Synthetic Data Predictions:", predictions)
    plot_decision_boundary(X_synthetic, y_synthetic, perceptron, "Perceptron Decision Boundary (Synthetic Data)")

# === Test 2: Real Breast Cancer Data === #
def real_data_test():
    print("\nRunning Real Data Test...")
    data = load_breast_cancer()
    X = data.data[:, :2]  # Use first two features
    y = data.target       # Binary labels: 0 (malignant), 1 (benign)

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X_train, y_train)

    predictions = perceptron.predict(X_test)
    print("Real Data Test Predictions:", predictions)
    plot_decision_boundary(X, y, perceptron, "Perceptron Decision Boundary (Breast Cancer Data)")

# Run Both Tests
if __name__ == "__main__":
    synthetic_test()
    real_data_test()
