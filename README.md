# INDE_577_Final_Project
Final project repository for INDE 577 at Rice University 

## Overview
This repository contains implementations of supervised and unsupervised machine learning algorithms for the **INDE 577** course at Rice University. Each algorithm is tested using both synthetic and real datasets, ensuring robustness and understanding of their behavior in different scenarios.

## Project Structure

```plaintext
INDE_577_Final_Project/
│
├── Supervised_Learning/
│   ├── boosting.py
│   ├── decision_trees.py
│   ├── knn.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── neural_networks.py
│   ├── perceptron.py
│   ├── random_forest.py
│   └── README.md
│
├── Unsupervised_Learning/
│   ├── dbscan.py
│   ├── kmeans.py
│   ├── pca.py
│   ├── svd_image_compression.py
│   └── README.md
│
├── requirements.txt
└── README.md  # Main Documentation

```

## Running the Project

1. Setting up the Virtual Environment
To set up and activate the virtual environment:

```bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

2. Running the Scripts
For Supervised Learning:

Navigate to the Supervised_Learning directory and run the desired script. Example:

```bash

# Run Perceptron
python perceptron.py

# Run Linear Regression
python linear_regression.py

```

For Unsupervised Learning:

Navigate to the Unsupervised_Learning directory and run the desired script. Example:

```bash

# Run KMeans Clustering
python kmeans.py

# Run SVD Image Compression
python svd_image_compression.py

```

## Results 

1. Supervised Learning Results:

Synthetic Data: Decision boundaries and regression lines are plotted for synthetic data.
Real Data: Results are tested on the Breast Cancer dataset and visualized.

2. Unsupervised Learning Results:

DBSCAN & KMeans: Clusters are visualized, with DBSCAN using PCA for real data.
PCA & SVD: Principal components are plotted, and image compression results are displayed.

## Notes

Ensure all dependencies are installed using requirements.txt:

```bash

pip install -r requirements.txt

```

Plots will display sequentially. Close the plot window to proceed to the next step.

## Contact

If you encounter issues or need further clarification, feel free to ask! Email: alex13brock@gmail.com

## Acknowledgment

This project is submitted as part of the INDE 577 course at Rice University.
