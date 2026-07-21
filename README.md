# methane-storage-machine-learning

Python workflow for machine learning-based methane storage suitability assessment in saline and depleted gas reservoirs.

## Overview

This repository contains the computational workflow developed for evaluating underground methane storage suitability using machine learning approaches. The framework integrates reservoir parameters, statistical analysis, machine learning prediction, and model performance evaluation to identify controlling factors affecting methane storage potential.

The workflow supports reproducible analysis of geological and reservoir properties including porosity, permeability, reservoir thickness, pressure, temperature, gas saturation, and other relevant storage parameters.

## Workflow Structure

The complete workflow includes:

1. Reservoir parameter assessment
2. Data preprocessing and machine learning prediction
3. Feature importance evaluation
4. Correlation analysis
5. Model accuracy comparison
6. Regression performance evaluation


## Repository Structure

```text
methane-storage-machine-learning/

│
├── scripts/
│   ├── reservoir_assessment.py
│   ├── machine_learning_prediction.py
│   ├── feature_importance_analysis.py
│   ├── correlation_matrix.py
│   ├── classification_accuracy.py
│   ├── r2_model_comparison.py
│   └── rmse_model_comparison.py
│
├── requirements.txt
├── LICENSE
└── README.md
```


## Description of Python Scripts

### reservoir_assessment.py

Performs initial reservoir suitability assessment and visualization of geological parameters controlling methane storage potential.


### machine_learning_prediction.py

Implements machine learning models for methane storage prediction.

Models included:

- Random Forest
- XGBoost
- Support Vector Machine (SVM)

The script performs:

- Data preprocessing
- Model training
- Prediction
- Performance evaluation


### feature_importance_analysis.py

Analyzes the contribution of individual reservoir parameters and identifies the dominant factors affecting methane storage suitability.


### correlation_matrix.py

Generates Pearson correlation analysis and heatmap visualization to investigate relationships among reservoir parameters.


### classification_accuracy.py

Compares classification accuracy among different machine learning models.


### r2_model_comparison.py

Evaluates regression performance using the coefficient of determination (R²).


### rmse_model_comparison.py

Calculates and compares Root Mean Square Error (RMSE) values for regression models.


## Installation

Clone this repository:

```bash
git clone https://github.com/Mazahir7327/methane-storage-machine-learning.git
```

Navigate to the repository folder:

```bash
cd methane-storage-machine-learning
```

Install required Python dependencies:

```bash
pip install -r requirements.txt
```


## Required Python Environment

The workflow was developed using Python 3.x.

Required packages:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost


## Running the Workflow

The scripts can be executed individually from the `scripts` directory.


### Reservoir Assessment

```bash
python scripts/reservoir_assessment.py
```


### Machine Learning Prediction

```bash
python scripts/machine_learning_prediction.py
```


### Feature Importance Analysis

```bash
python scripts/feature_importance_analysis.py
```


### Correlation Analysis

```bash
python scripts/correlation_matrix.py
```


### Classification Accuracy Evaluation

```bash
python scripts/classification_accuracy.py
```


### R² Model Comparison

```bash
python scripts/r2_model_comparison.py
```


### RMSE Model Comparison

```bash
python scripts/rmse_model_comparison.py
```


## Output Results

The workflow generates:

- Reservoir parameter visualization
- Machine learning prediction results
- Feature importance plots
- Correlation heatmaps
- Classification performance evaluation
- Regression model comparison results


## Reproducibility

All Python scripts required to reproduce the machine learning analysis and visualization results presented in the manuscript are provided in this repository.

The workflow can be adapted for different geological storage scenarios by modifying input reservoir parameters and machine learning settings.


## Scientific Application

This repository provides a reproducible computational framework for methane storage suitability assessment by integrating:

- Geological parameter analysis
- Machine learning prediction
- Statistical evaluation
- Model comparison


## License

This project is released under the MIT License.
