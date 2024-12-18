import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def synthetic_test():
    """Synthetic test for PCA using generated classification data."""
    print("Running Synthetic Test...\n")

    # Generate synthetic data
    X, _ = make_classification(n_samples=300, n_features=5, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolors='k')
    plt.title("PCA Projection - Synthetic Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # Print explained variance ratio
    print(f"Explained Variance Ratio (Synthetic Data): {pca.explained_variance_ratio_}")

def real_data_test():
    """Real data test for PCA using Breast Cancer dataset."""
    print("\nRunning Real Data Test...")

    # Load Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target  # For visualization, target classes (0, 1)

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot first two principal components, colored by class
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title("PCA Projection - Breast Cancer Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Class (0: Malignant, 1: Benign)")
    plt.show()

    # Print explained variance ratio
    print(f"Explained Variance Ratio (Real Data): {pca.explained_variance_ratio_}")

if __name__ == "__main__":
    synthetic_test()
    real_data_test()
