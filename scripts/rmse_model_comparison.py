"""
RMSE Model Comparison for Methane Storage Prediction

This script compares regression models using
Root Mean Square Error (RMSE).

Models:
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regression

Author:
Mazahir Hussain
"""


import os
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import numpy as np



# Dataset path

DATA_FILE = "../data/methane_storage_dataset.csv"



# Load dataset

def load_data():

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip()

    return df



# Compare models

def compare_models(df):


    target = "Storage Capacity"


    X = df.drop(
        columns=[target]
    )

    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )



    models = {

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42
            ),


        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                random_state=42
            ),


        "SVR":
            SVR(
                kernel="rbf"
            )

    }



    results = {}



    for name, model in models.items():


        model.fit(
            X_train,
            y_train
        )


        prediction = model.predict(
            X_test
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )


        results[name] = rmse



    return results



# Plot RMSE comparison

def plot_results(results):


    os.makedirs(
        "../results/generated_figures",
        exist_ok=True
    )


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        results.keys(),
        results.values()
    )


    plt.ylabel(
        "RMSE"
    )


    plt.xlabel(
        "Machine Learning Model"
    )


    plt.title(
        "RMSE Model Comparison"
    )


    plt.tight_layout()



    plt.savefig(
        "../results/generated_figures/rmse_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# Main execution

if __name__ == "__main__":


    data = load_data()


    results = compare_models(
        data
    )


    print(results)


    os.makedirs(
        "../results",
        exist_ok=True
    )


    pd.DataFrame(
        results.items(),
        columns=[
            "Model",
            "RMSE"
        ]
    ).to_csv(
        "../results/rmse_model_comparison.csv",
        index=False
    )


    plot_results(
        results
    )


    print(
        "RMSE comparison completed."
    )
