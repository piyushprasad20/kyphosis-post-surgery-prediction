# kyphosis-post-surgery-prediction


## Overview
This project builds and compares tree-based supervised machine learning models to predict the recurrence of **kyphosis** (an abnormal convex curvature of the spine) in pediatric patients following corrective spinal surgery. Using clinical surgical records, the workflow evaluates how effectively a single Decision Tree and an ensemble Random Forest classifier can distinguish between successful recoveries and post-operative spinal deformities.

---

## Dataset Summary
The analysis uses the clinical `kyphosis.csv` dataset, which tracks **81 pediatric surgical cases** across four key medical attributes:

| Feature / Target | Role | Description |
| :--- | :--- | :--- |
| **`Age`** | Input Feature | Age of the child at the time of surgery (measured in months). |
| **`Number`** | Input Feature | Total number of vertebrae involved in the surgical procedure. |
| **`Start`** | Input Feature | Index of the topmost (first) vertebra operated on. |
| **`Kyphosis`** | Target Variable | Indicates whether the spinal deformity was **`present`** or **`absent`** after surgery. |

---

## Workflow & Methodology

1. **Exploratory Data Analysis (EDA):**
   * Generated multi-variable pairplots grouped by surgical outcome (`Kyphosis`) to inspect feature distributions, correlations, and class separation across `Age`, `Number`, and `Start`.
2. **Data Preparation:**
   * Separated clinical predictors (`X`) from the diagnostic outcome (`y`) and partitioned the records into a **70% training set** and a **30% holdout test set**.
3. **Model Training & Tree Visualization:**
   * Trained a baseline `DecisionTreeClassifier` using Gini impurity splits.
   * Exported and rendered the internal decision rules as a visual flowchart using `graphviz` and `pydot` to inspect exact threshold splits.
4. **Ensemble Comparison:**
   * Trained a `RandomForestClassifier` configured with **100 estimators** (`n_estimators=100`) to aggregate predictions across multiple randomized trees and reduce single-tree variance.

---

## Model Performance Comparison

Both models were evaluated on the unseen test split (25 patients: 20 `absent`, 5 `present`):

| Metric | Decision Tree | Random Forest (100 Trees) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 76% | **80%** |
| **Precision (`absent`)** | 0.85 | **0.86** |
| **Recall (`absent`)** | 0.85 | **0.90** |
| **Precision (`present`)** | 0.40 | **0.50** |
| **F1-Score (`present`)** | 0.40 | **0.44** |

### Key Takeaways
* **Ensemble Advantage:** The Random Forest model outperformed the single Decision Tree in overall accuracy (80% vs. 76%) and precision on positive cases by reducing false positives.
* **Class Imbalance Impact:** Because `absent` cases significantly outnumber `present` cases in the dataset, both models achieve strong recall on negative outcomes while showing lower sensitivity on minority `present` cases.

---

## Tech Stack
* **Python 3**
* **Data Handling:** `pandas`, `numpy`
* **Visualization:** `seaborn`, `matplotlib`, `pydot`, `graphviz`
* **Machine Learning:** `scikit-learn`
