"""
Feature Importance Analysis for Methane Storage Suitability

This script calculates and visualizes the importance
of geological parameters using Random Forest.

Author:
Mazahir Hussain

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



# =====================================================
# Data path
# =====================================================

DATA_FILE = "../data/methane_storage_dataset.csv"



# =====================================================
# Load dataset
# =====================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip()

    return df



# =====================================================
# Feature importance calculation
# =====================================================

def calculate_feature_importance(df):


    target = "Storage Suitability"


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


    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }
    )


    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )


    return importance



# =====================================================
# Plot feature importance
# =====================================================

def plot_importance(importance):


    plt.figure(
        figsize=(10,6)
    )


    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )


    plt.xlabel(
        "Feature Importance"
    )


    plt.ylabel(
        "Reservoir Parameters"
    )


    plt.title(
        "Machine Learning Feature Importance"
    )


    plt.gca().invert_yaxis()


    os.makedirs(
        "../results/generated_figures",
        exist_ok=True
    )


    plt.savefig(
        "../results/generated_figures/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    data = load_data()


    importance = calculate_feature_importance(
        data
    )


    print(
        importance
    )


    os.makedirs(
        "../results",
        exist_ok=True
    )


    importance.to_csv(
        "../results/feature_importance.csv",
        index=False
    )


    plot_importance(
        importance
    )


    print(
        "Feature importance analysis completed."
    )
