"""
Machine Learning Prediction for Methane Storage Suitability Assessment
---------------------------------------------------------------------

This script performs machine learning analysis using:

- Random Forest
- XGBoost
- Support Vector Machine

The workflow includes:
- Data preprocessing
- Model training
- Prediction
- Performance evaluation
- Feature importance analysis

Author:
Mazahir Hussain

"""


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor



# ==========================================================
# File location
# ==========================================================

DATA_FOLDER = "../data"

data_file = os.path.join(
    DATA_FOLDER,
    "methane_storage_dataset.csv"
)



# ==========================================================
# Load data
# ==========================================================

def load_dataset():

    df = pd.read_csv(data_file)

    df.columns = df.columns.str.strip()

    print("\nDataset Loaded")
    print(df.head())

    return df



# ==========================================================
# Data preparation
# ==========================================================

def prepare_data(df):

    target_column = "Storage Suitability"

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_test_scaled = scaler.transform(
        X_test
    )


    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X.columns
    )



# ==========================================================
# Model evaluation
# ==========================================================

def evaluate_model(name, model, X_test, y_test):

    prediction = model.predict(
        X_test
    )


    r2 = r2_score(
        y_test,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    mae = mean_absolute_error(
        y_test,
        prediction
    )


    print("\n", name)

    print(
        "R2:",
        round(r2,4)
    )

    print(
        "RMSE:",
        round(rmse,4)
    )

    print(
        "MAE:",
        round(mae,4)
    )


    return prediction



# ==========================================================
# Main Machine Learning Workflow
# ==========================================================


if __name__ == "__main__":


    df = load_dataset()


    (
        X_train,
        X_test,
        y_train,
        y_test,
        features

    ) = prepare_data(df)



    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )


    rf_model.fit(
        X_train,
        y_train
    )


    evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test
    )



    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )


    xgb_model.fit(
        X_train,
        y_train
    )


    evaluate_model(
        "XGBoost",
        xgb_model,
        X_test,
        y_test
    )



    # ------------------------------------------------------
    # Support Vector Machine
    # ------------------------------------------------------

    svm_model = SVR(
        kernel="rbf"
    )


    svm_model.fit(
        X_train,
        y_train
    )


    evaluate_model(
        "Support Vector Machine",
        svm_model,
        X_test,
        y_test
    )



    # ------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------

    importance = pd.DataFrame(
        {
            "Feature": features,
            "Importance": rf_model.feature_importances_
        }
    )


    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )


    print("\nFeature Importance")
    print(importance)



    # Save results

    os.makedirs(
        "../results",
        exist_ok=True
    )


    importance.to_csv(
        "../results/feature_importance.csv",
        index=False
    )


    print(
        "\nMachine learning analysis completed successfully."
    )
