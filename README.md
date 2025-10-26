# 🩺 Heart Disease Prediction and Risk Profiling with PySpark

### End-to-End Data Science Project – Exploratory Analysis, Feature Engineering, Supervised Models, and Unsupervised Clustering

---

## 📘 Overview
This project explores the **Heart Disease dataset** using **Apache Spark (PySpark)** for scalable data processing and machine learning.  
The goal is twofold:
1. **Predictive Modeling** – Build and compare supervised ML models to classify patients as having heart disease (1) or not (0).  
2. **Phenotype Discovery** – Apply unsupervised clustering to uncover natural patient subgroups based on ECG patterns, symptoms, and clinical measurements.

The analysis integrates **data engineering, feature preprocessing, model training, and interpretation**, representing a full data science workflow optimized for large-scale health analytics.

---

## 🧠 Key Objectives

- Perform robust **data cleaning and exploratory data analysis (EDA)** on clinical variables.
- Engineer interpretable and ML-ready features from mixed numerical and categorical data.
- Train and evaluate multiple **supervised models**:
  - Logistic Regression (Elastic Net)
  - Decision Tree Classifier
  - Random Forest
  - Gradient Boosted Trees (GBT)
- Visualize and compare model performance (ROC curves, feature importances).
- Conduct **unsupervised K-Means clustering** to identify risk-based patient groups.
- Interpret results from both a **data-scientific** and **cardiology** perspective.

---

## 🧩 Dataset Description

The dataset contains **918 patient records**, each describing demographic, symptomatic, and ECG-related features.

| Feature | Type | Description |
|:--|:--|:--|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | M / F |
| `ChestPainType` | Categorical | ATA, NAP, ASY, TA |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dl) |
| `FastingBS` | Numeric | Fasting blood sugar (>120 mg/dl → 1, else 0) |
| `RestingECG` | Categorical | Normal, LVH, ST |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Y / N |
| `Oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `ST_Slope` | Categorical | Up, Flat, Down |
| `HeartDisease` | Target | 1 = presence, 0 = absence of heart disease |

---

## ⚙️ Pipeline & Methodology

### 1️⃣ Data Preparation
- Load CSV using `spark.read.csv` with schema inference and null handling.
- Type casting and normalization of continuous features.
- Encoding of categorical variables using `StringIndexer` + `OneHotEncoder`.
- Combined all features into a single vector via `VectorAssembler`.

### 2️⃣ Exploratory Data Analysis (EDA)
- Statistical summaries and missing value checks.
- Correlation analysis between continuous features and target variable.
- Feature distribution and pairwise visualization.
- Categorical frequency analysis (e.g., Chest Pain Type vs. Heart Disease).

### 3️⃣ Supervised Modeling
- **Baseline:** Logistic Regression (ElasticNet regularization)  
- **Tree-Based:** Decision Tree, Random Forest, Gradient Boosted Trees  
- Models evaluated via:
  - Accuracy, Precision, Recall, F1-score
  - ROC–AUC
- Visual diagnostics:
  - ROC curves for all models  
  - Feature importance plots (Logistic Coefficients & Tree Importances)


### 4️⃣ Unsupervised Clustering
- Applied **K-Means** on the full feature vector (`features`).
- Selected optimal number of clusters `k=3` via **silhouette score (0.303)**.
- PCA (2D) visualization of cluster structure.
- Cluster-level feature summaries and clinical interpretation.

---

## 📊 Results Summary

### 🔸 Supervised Models
- **Gradient Boosted Trees (GBT)** achieved the best trade-off between recall and precision.  
- Feature importances consistently highlighted:
  - `ST_Slope`  
  - `Oldpeak`  
  - `MaxHR`  
  - `ExerciseAngina`  
  - `ChestPainType`

### 🔸 Clustering Analysis (k=3)
| Cluster | Size | Pos. Rate | Avg Age | MaxHR | Oldpeak | Interpretation |
|:--|:--|:--|:--|:--|:--|:--|
| **1** | 401 | 0.18 | 48 | 153 | 0.27 | Healthy / low-risk group |
| **0** | 332 | 0.82 | 58 | 126 | 1.7 | Ischemic high-risk group |
| **2** | 185 | 0.89 | 57 | 120 | 0.8 | Chronic / very high-risk group |

### 🔸 Key Insights
- Unsupervised clusters align closely with **clinical risk phenotypes**:
  - `ST_Slope` and `Oldpeak` dominate ECG-based separation.
  - Low-risk patients: upsloping ST, no angina, high MaxHR.
  - High-risk patients: flat/downsloping ST, exercise-induced angina, reduced MaxHR.
- Confirms model interpretability and dataset reliability.

---
