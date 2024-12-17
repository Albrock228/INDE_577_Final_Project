import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def synthetic_test():
    """Synthetic test for DBSCAN with make_moons dataset."""
    print("Running Synthetic Test...\n")

    # Generate synthetic data
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise
            color = 'k'
            label_name = "Noise"
        else:
            color = plt.cm.jet(float(label) / max(unique_labels))
            label_name = f"Cluster {label}"
        plt.scatter(X[labels == label, 0], X[labels == label, 1], color=color, label=label_name, edgecolors='k')

    plt.title("DBSCAN Clustering - Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def real_data_test():
    """Real data test for DBSCAN using Breast Cancer dataset with PCA."""
    print("\nRunning Real Data Test...")

    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_pca)

    # Print the cluster assignments
    print(f"Cluster Labels:\n{labels}")

    # Plot results
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise
            color = 'k'
            label_name = "Noise"
        else:
            color = plt.cm.jet(float(label) / max(unique_labels))
            label_name = f"Cluster {label}"
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], color=color, label=label_name, edgecolors='k')

    plt.title("DBSCAN Clustering - Breast Cancer Data with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    synthetic_test()
    real_data_test()
