# Supervised Learning

This folder contains implementations of various supervised learning algorithms. Each script includes:
1. A **synthetic data test** for quick validation and visualization.
2. A **real data test** using the **Breast Cancer dataset** (binary classification) and other datasets where applicable.

---

## Algorithms Implemented

### 1. **Linear Regression**  
- **Purpose**: Predicts a continuous target variable using a linear relationship.
- **Tests**:
   - **Synthetic Test**: Simple 1D dataset.
   - **Real Test 1**: Breast Cancer dataset (binary target) to showcase limitations.
   - **Real Test 2**: California Housing dataset (continuous target) for proper regression.

**Script**: `linear_regression.py`

---

### 2. **Logistic Regression**  
- **Purpose**: A linear classifier for binary classification problems.
- **Tests**:
   - **Synthetic Test**: Separates two classes with a straight line.
   - **Real Test**: Breast Cancer dataset for binary classification.

**Script**: `logistic_regression.py`

---

### 3. **Perceptron**  
- **Purpose**: A simple linear binary classifier inspired by neural networks.
- **Tests**:
   - **Synthetic Test**: Separates two classes using a straight decision boundary.
   - **Real Test**: Breast Cancer dataset.

**Script**: `perceptron.py`

---

### 4. **K-Nearest Neighbors (KNN)**  
- **Purpose**: Classifies points based on the majority label of their nearest neighbors.
- **Tests**:
   - **Synthetic Test**: Uses KNN for class separation.
   - **Real Test**: Breast Cancer dataset.

**Script**: `knn.py`

---

### 5. **Decision Trees**  
- **Purpose**: A non-linear classifier that partitions the data using axis-aligned splits.
- **Tests**:
   - **Synthetic Test**: Displays decision regions.
   - **Real Test**: Breast Cancer dataset.

**Script**: `decision_trees.py`

---

### 6. **Random Forest**  
- **Purpose**: An ensemble of decision trees that reduces overfitting.
- **Tests**:
   - **Synthetic Test**: Uses 10 trees with depth=2.
   - **Real Test**: Breast Cancer dataset.

**Script**: `random_forest.py`

---

### 7. **Boosting (AdaBoost)**  
- **Purpose**: An ensemble method that combines weak classifiers to improve performance.
- **Tests**:
   - **Synthetic Test**: Uses shallow decision trees (depth=1).
   - **Real Test**: Breast Cancer dataset.

**Script**: `boosting.py`

---

### 8. **Neural Networks (MLP)**  
- **Purpose**: A feedforward neural network for binary classification.
- **Tests**:
   - **Synthetic Test**: Demonstrates a non-linear decision boundary.
   - **Real Test**: Breast Cancer dataset.

**Script**: `neural_networks.py`

---

## Testing Framework

Each script performs the following tests:
1. **Synthetic Test**:
   - Small, simple datasets for visualization.
   - Helps validate algorithm functionality.

2. **Real Data Test**:
   - Uses the **Breast Cancer dataset** from `sklearn.datasets` for binary classification.
   - Linear Regression includes an additional test using the **California Housing dataset**.

---

## How to Run

To run any script, activate the virtual environment and execute the Python script. For example:

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run Linear Regression
python linear_regression.py

# Run Logistic Regression
python logistic_regression.py
```

## Results

1. Synthetic Data:
   - Decision boundaries are visualized for classification models.
   - Regression models display the regression line.

2. Real Data:
   - Decision boundaries are plotted for the first two features of the Breast Cancer dataset.
   - Predictions are printed for verification.

## Notes

- Ensure all dependencies are installed using the following command:
    pip install -r requirements.txt

- Plots will display sequentially. Close the plot window to proceed to the next step.

## Contact

If you encounter issues or need further clarification, feel free to ask! Email: alex13brock@gmail.com
