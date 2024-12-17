import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def synthetic_test():
    """Synthetic test for K-Means clustering with make_blobs dataset."""
    print("Running Synthetic Test...\n")

    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title("K-Means Clustering - Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def real_data_test():
    """Real data test for K-Means clustering using Breast Cancer dataset."""
    print("\nRunning Real Data Test...")

    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    # Print cluster assignments
    print(f"Cluster Labels (First 20):\n{labels[:20]}")

    # Plot results
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f"Cluster {i}")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title("K-Means Clustering - Breast Cancer Data with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    synthetic_test()
    real_data_test()
