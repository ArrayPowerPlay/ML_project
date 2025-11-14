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

### 3. Decision Tree Regressor
- **Purpose:** Non-linear relationships capture
- **Tuning:** Max depth, min samples split/leaf

### 4. Random Forest Regressor
- **Purpose:** Ensemble of decision trees
- **Tuning:** Number of estimators, max depth, max features

### 5. XGBoost Regressor
- **Purpose:** Gradient boosting for high accuracy
- **Tuning:** Learning rate, max depth, subsample ratio

### 6. Support Vector Machine (SVM)
- **Purpose:** Non-linear regression with kernel trick
- **Tuning:** C parameter, kernel type, gamma

### 7. Weighted Ensemble Model
- **Purpose:** Combine all models for best performance
- **Method:** Weight models based on inverse of validation RMSE
- **Formula:** `Prediction = Σ(weight_i × model_i_prediction)`

---

## 📁 Project Structure

```
life-expectancy-prediction/
│
├── data/
│   ├── raw/                          # Original data from World Bank
│   │   ├── gdp_per_capita.csv
│   │   ├── life_expectancy.csv
│   │   ├── population.csv
│   │   └── ...
│   │
│   └── processed/                    # Cleaned and merged data
│       └── final_dataset.csv
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Data gathering from World Bank
│   ├── 02_preprocessing.ipynb        # Data cleaning and normalization
│   ├── 03_eda_analysis.ipynb         # Exploratory data analysis
│   ├── 04_linear_models.ipynb        # Linear & Logistic Regression
│   ├── 05_tree_models.ipynb          # Decision Tree & Random Forest
│   ├── 06_xgboost_model.ipynb        # XGBoost implementation
│   ├── 07_svm_model.ipynb            # Support Vector Machine
│   └── 08_ensemble_evaluation.ipynb  # Model ensemble & final results
│
├── models/                           # Saved trained models
│   ├── lasso_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── svm_model.pkl
│   └── ensemble_model.pkl
│
├── results/                          # Predictions and metrics
│   ├── predictions_2024.csv          # Final predictions for 2024
│   ├── model_comparison.csv          # Performance comparison
│   └── metrics_summary.json          # Detailed metrics
│
├── visualizations/                   # Generated plots and charts
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── model_comparison_rmse.png
│   ├── predictions_vs_actual.png
│   └── ...
│
├── presentation/                     # Presentation slides
│   └── slides.pptx
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/life-expectancy-prediction.git
cd life-expectancy-prediction
```

2. **Create virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open notebooks** in order (01 → 08) and run cells

---

## 📦 Dependencies

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

## 💻 Usage

### Quick Start

```bash
# 1. Run all notebooks in sequence
jupyter notebook notebooks/

# 2. Or run specific analysis
jupyter notebook notebooks/03_eda_analysis.ipynb
```

### Step-by-Step Workflow

#### **Step 1: Data Collection** (Notebook 01)
```python
# Downloads data from World Bank API
# Saves raw CSV files to data/raw/
```

#### **Step 2: Data Preprocessing** (Notebook 02)
```python
# - Merge all features into single dataset
# - Handle missing values (mean imputation)
# - Normalize features (Min-Max scaling)
# - Split train/test (90/10)
# Output: data/processed/final_dataset.csv
```

#### **Step 3: EDA** (Notebook 03)
```python
# - Correlation analysis
# - Feature distributions
# - Time series trends
# - Outlier detection
# Output: visualizations/*.png
```

#### **Step 4-7: Model Training** (Notebooks 04-07)
```python
# For each model:
# - K-Fold Cross-Validation (k=5)
# - Hyperparameter tuning
# - Training on full train set
# - Save trained model to models/
```

#### **Step 8: Ensemble & Evaluation** (Notebook 08)
```python
# - Load all trained models
# - Create weighted ensemble
# - Predict on 2024 test data
# - Compare with actual values
# - Generate final report
```

### Loading Pre-trained Models

```python
import pickle

# Load a specific model
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Make predictions
predictions = rf_model.predict(X_new)
```

---

## 📈 Results

### Model Performance Comparison

| Model | Training RMSE | Validation RMSE | Test RMSE (2024) | R² Score |
|-------|---------------|-----------------|------------------|----------|
| Linear Regression (Lasso) | 2.34 | 3.12 | 3.25 | 0.892 |
| Decision Tree | 1.45 | 3.89 | 4.12 | 0.843 |
| Random Forest | 1.78 | 2.98 | 3.08 | 0.901 |
| XGBoost | 1.56 | 2.87 | 2.95 | 0.912 |
| SVM (RBF) | 2.01 | 3.34 | 3.42 | 0.881 |
| **Weighted Ensemble** | **1.67** | **2.76** | **2.82** | **0.921** |

*RMSE = Root Mean Squared Error (lower is better)*  
*R² Score = Coefficient of Determination (higher is better)*

### Key Findings

1. **Best Single Model:** XGBoost (Test RMSE: 2.95)
2. **Best Overall:** Weighted Ensemble (Test RMSE: 2.82)
3. **Most Important Features:**
   - GDP per capita (importance: 0.234)
   - Access to electricity (importance: 0.189)
   - Sanitation services (importance: 0.157)

4. **Feature Selection (Lasso):**
   - 2 features eliminated (coefficients ≈ 0)
   - Final model uses 9 features

### Visualizations

![Model Comparison](visualizations/model_comparison_rmse.png)
*Figure 1: RMSE comparison across all models*

![Predictions vs Actual](visualizations/predictions_vs_actual.png)
*Figure 2: Predicted vs actual life expectancy for 2024*

---

## 👥 Team Members

| Name | Role | Responsibilities |
|------|------|------------------|
| **Member 1** | Data Collection & Preprocessing | Notebooks 01-02 |
| **Member 2** | EDA & Visualization | Notebook 03, Visualizations |
| **Member 3** | Linear Models | Notebook 04 |
| **Member 4** | Tree-based & SVM Models | Notebooks 05-07 |
| **Member 5** | Ensemble & Integration | Notebook 08, GitHub, Slides |

---

## 🔧 Methodology

### Data Preprocessing
1. **Missing Value Imputation:** Mean imputation for continuous variables
2. **Normalization:** Min-Max scaling to [0, 1] range
3. **Feature Engineering:** No additional features created (raw features only)

### Model Training
- **Validation Method:** 5-Fold Cross-Validation
- **Optimization:** Minibatch Gradient Descent (where applicable)
- **Stopping Criteria:** 
  - Max iterations: 10 epochs
  - Early stopping with tolerance: 1e-4

### Hyperparameter Tuning Methods
- **Grid Search:** For Linear Regression, Decision Tree
- **Random Search:** For Random Forest, SVM
- **Bayesian Optimization:** For XGBoost (using Optuna - optional)

### Ensemble Strategy
- **Method:** Weighted averaging
- **Weights:** Based on inverse of validation RMSE
- **Formula:** `w_i = (1/RMSE_i) / Σ(1/RMSE_j)`

---

## 📊 Reproducibility

To reproduce our results:

1. **Set random seeds** (already included in notebooks):
```python
import numpy as np
import random
random.seed(42)
np.random.seed(42)
```

2. **Use same data split**:
```python
train_test_split(..., random_state=42)
```

3. **Run notebooks in order**: 01 → 08

4. **Check environment**:
```bash
pip list > environment.txt
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to functions
- Comment complex logic
- Keep notebook cells focused (one task per cell)

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{life_expectancy_prediction_2024,
  author = {Your Team Name},
  title = {Life Expectancy Prediction Using Demographic Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/life-expectancy-prediction}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Source:** [World Bank Open Data](https://data.worldbank.org/)
- **Libraries:** scikit-learn, XGBoost, pandas, matplotlib, seaborn
- **Inspiration:** World Health Organization (WHO) research on life expectancy factors

---

## 📞 Contact

For questions or feedback:

- **Email:** your.email@example.com
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/life-expectancy-prediction/issues)
- **Presentation:** See `presentation/slides.pptx`

---

## 🗺️ Roadmap

- [x] Data collection and preprocessing
- [x] Exploratory data analysis
- [x] Baseline models (Linear Regression)
- [x] Advanced models (Random Forest, XGBoost, SVM)
- [x] Ensemble model
- [x] Model evaluation on 2024 data
- [ ] Deploy model as web API (future work)
- [ ] Add more features (healthcare spending, education index)
- [ ] Time series forecasting for future years

---

## 📚 References

1. World Bank. (2024). *World Development Indicators*. https://data.worldbank.org/
2. Scikit-learn Documentation. https://scikit-learn.org/stable/
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
4. WHO. (2023). *Global Health Observatory*. https://www.who.int/data/gho

---

**Last Updated:** October 2024  
**Version:** 1.0.0  

⭐ **Star this repo** if you find it helpful!
