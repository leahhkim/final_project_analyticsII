# DS4021 Final Project - Infant Mortality Prediction

- Team Members: Ethan Cao, Leah Kim, Eden Mulugeta

## Overview

Infant mortality remains a critical public health concern, with approximately 20,000 deaths occurring in the United States annually among infants under one year of age. This project applies machine learning techniques to predict infant mortality risk using comprehensive birth and health data from the National Vital Statistics System (NVSS). By identifying high-risk cases early, this work aims to support targeted interventions that can improve neonatal outcomes and reduce preventable infant deaths.

The analysis leverages a diverse set of maternal, prenatal, and neonatal factors including demographic information, prenatal care patterns, maternal health conditions, gestational characteristics, birth outcomes, and congenital anomalies. Multiple predictive models are developed and compared to identify the most effective approach for infant mortality prediction.

## Dataset

This project uses the NVSS Linked Birth-Infant Death dataset (2013), which combines birth certificate data with infant death records for comprehensive mortality analysis. The dataset includes:

- Over 3.9 million birth records
- Maternal demographics and health history
- Prenatal care utilization patterns
- Birth outcomes and neonatal interventions
- Congenital anomaly indicators
- Linked mortality outcomes for infants who died before age one

The data has been preprocessed to remove potential label leakage, handle missing values, and create balanced train/validation/test splits suitable for machine learning.

## Repository Structure

```text
.
├── README.md                              # Project overview and documentation
│
├── Data/                                  # Data files and preprocessing
│   ├── linkco2013us_den.csv              # Original NVSS linked birth-death dataset
│   ├── Train_Test_Validaiton_Split.ipynb # Data preprocessing and split generation
│   ├── nvss_train.csv                    # Training set (60% of data)
│   ├── nvss_val.csv                      # Validation set (20% of data)
│   ├── nvss_test.csv                     # Test set (20% of data)
│   └── nvss_aggregated.csv               # Complete preprocessed dataset
│
├── NOTEBOOKS/                             # Machine learning model development
│   ├── SVM.ipynb                         # Support Vector Machine implementation
│   ├── Neural_Network_Prediction.ipynb   # Neural network models
│   └── Ensemble.ipynb                    # Ensemble methods and model comparison
│
└── env/                                   # Python virtual environment
```

## Key Features Used for Prediction

The models utilize carefully selected features from the NVSS dataset:

**Maternal Demographics:** Age, education, race, ethnicity, marital status

**Obstetric History:** Live birth order, previous pregnancies, prior cesarean deliveries, history of preterm birth or poor outcomes

**Prenatal Care:** Month care began, number of prenatal visits, gestational weight gain

**Maternal Health Conditions:** Diabetes, chronic hypertension, pregnancy-associated hypertension, eclampsia

**Maternal Risk Behaviors:** Tobacco use during pregnancy, cigarettes per day

**Infant Characteristics:** Sex, gestational age, birth weight, 5-minute Apgar score, plurality (singleton/twin/etc.)

**Neonatal Interventions:** Assisted ventilation, NICU admission, extended ventilation

**Congenital Anomalies:** Neural tube defects, congenital heart disease, chromosomal disorders, and other major birth defects

## Requirements

**Python Version:** 3.10 or higher

**Required Packages:**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Usage

1. **Data Preprocessing:** Start with `Data/Train_Test_Validaiton_Split.ipynb` to understand the data preparation process
2. **Model Training:** Explore the notebooks in `NOTEBOOKS/` to see different modeling approaches
3. **Model Evaluation:** Compare model performance metrics across different algorithms

## Models Implemented

- Support Vector Machine (SVM)
- Neural Networks
- Ensemble Methods
- Regularized Logistic Regression

Each model is trained on the same preprocessed dataset and evaluated using consistent metrics including accuracy, precision, recall, F1-score, and ROC-AUC.

## Project Goals

1. Develop accurate predictive models for infant mortality risk
2. Identify the most important risk factors contributing to infant mortality
3. Provide interpretable results that can inform clinical decision-making
4. Compare multiple machine learning approaches to determine optimal methodology

## Contributors

This project was developed as part of DS4021 focused on applying machine learning to real-world healthcare challenges.
