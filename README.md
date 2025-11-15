# Life Expectancy Prediction Using Demographic Data

> Predicting life expectancy across 200+ countries using machine learning models trained on World Bank demographic data (2000-2024)

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Team Members](#team-members)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project uses supervised learning to predict **life expectancy** based on demographic and socioeconomic indicators from the World Bank. We train multiple machine learning models on data from 2000-2024 and evaluate their performance.

**Key Highlights:**
- **200+ countries** analyzed
- **20+ years** of historical data (2000-2024)
- **6+ ML algorithms** compared
- **Weighted ensemble** for final predictions
- **Comprehensive EDA** with visualizations

---

## Problem Statement

**Goal:** Predict life expectancy at birth for countries worldwide using demographic features.

**Type:** Supervised Learning (Regression)

**Approach:**
1. Train models on historical data (2000-2024)
2. Tune hyperparameters using K-Fold Cross-Validation
3. Choose the best model based on the validation set
4. Evaluate models on unseen test dataset 

---

## Dataset

### Data Source
- **World Bank Open Data** (https://data.worldbank.org/)
- **Time Period:** 2000-2024
- **Coverage:** 200+ countries
- **Features:** 10+ demographic and socioeconomic indicators

### Features Used

- **Population, total** 
- **Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)** 
- **Population growth (annual %)** 
- **GDP per capita (current US$)** 
- **GDP growth (annual %)**
- **People using safely managed sanitation services (% of population)**
- **Access to electricity (% of population)** 
- **People using at least basic drinking water services (% of population)**
- **Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)**
- **Population living in slums (% of urban population)** 
- **Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)**
- **Year**

**Target Variable:** Life expectancy at birth (years)

### Data Split
- **Training Set:** 60%
- **Validation Set:** 20% 
- **Test Set:** 20% 

---

## Models

We implemented and compared the following machine learning algorithms:

### 1. Linear Regression / Ridge / Lasso
- **Regularization:** L1 or L2 penalty to eliminate less important features
- **Hyperparameter Tuning:** K-Fold CV for optimal alpha

### 2. Support Vector Machine (SVM)
- **Purpose:** Non-linear regression with kernel trick
- **Tuning:** C parameter, kernel type, gamma

### 3. Decision Tree Regressor
- **Purpose:** Non-linear relationships capture
- **Tuning:** Max depth, min samples split/leaf

### 4. Random Forest Regressor
- **Purpose:** Ensemble of decision trees
- **Tuning:** Number of estimators, max depth, max features

### 5. Gradient Boosting Machine
- **Purpose:** Sequential boosting where each tree corrects errors from the previous one
- **Tuning:** Learning rate, number of estimators, max depth

### 6. LightGBM Regressor
- **Purpose:** Optimized gradient boosting using leaf-wise tree growth
- **Tuning:** Number of leaves, learning rate, max depth

### 7. CatBoost Regressor
- **Purpose:** Gradient boosting designed to handle categorical features automatically
- **Tuning:** Depth, learning rate, iterations, bagging temperature
  
### 8. XGBoost Regressor
- **Purpose:** Gradient boosting for high accuracy
- **Tuning:** Learning rate, max depth, subsample ratio

### 9. Weighted Ensemble Model
- **Purpose:** Combine all models for best performance
- **Method:**
  1. Train several base models
  2. Use their predictions as input features for a meta-learner 
  3. Meta-learner learns how to best combine predictions

---

## Dependencies

```txt
# Core
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Machine Learning
xgboost>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
jupyter>=1.0.0
ipykernel>=6.25.0
joblib>=1.3.0
tqdm>=4.66.0

# Optional
jupytext>=1.15.0  # For .py ↔ .ipynb sync
```
---

## Team Members

- **Trần Gia Định** (team leader)
- **Nguyễn Quang Đức**
- **Hồ Minh Dũng**
- **Nguyễn Bá Đức Anh**
- **Lê Đức Chính**
  
---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- **Data Source:** [World Bank Open Data](https://data.worldbank.org/)
- **Libraries:** scikit-learn, XGBoost, pandas, matplotlib, seaborn
- **Inspiration:** World Health Organization (WHO) research on life expectancy factors

---

## Roadmap

- [x] Data collection and preprocessing
- [x] Exploratory data analysis and data visualization
- [x] Baseline models (Linear Regression)
- [x] Advanced models (Random Forest, XGBoost, SVM)
- [x] Ensemble model
- [x] Model selection based on validation data 
- [x] Model evaluation on unseen test data

---

## References

1. World Bank. (2024). *World Development Indicators*. https://data.worldbank.org/
2. Scikit-learn Documentation. https://scikit-learn.org/stable/
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
4. WHO. (2023). *Global Health Observatory*. https://www.who.int/data/gho

---

**Last Updated:** November 2024  
**Version:** 1.0.0  

⭐ **Star this repo** if you find it helpful!
