"""
Classification Accuracy Evaluation for Methane Storage Prediction

This script evaluates classification models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Author:
Mazahir Hussain
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Dataset path

DATA_FILE = "../data/methane_storage_dataset.csv"



# Load dataset

def load_data():

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip()

    return df



# Train classification model

def train_classifier(df):


    target = "Storage Class"


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


    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    return y_test, predictions



# Save confusion matrix

def plot_confusion_matrix(y_test, predictions):


    cm = confusion_matrix(
        y_test,
        predictions
    )


    os.makedirs(
        "../results/generated_figures",
        exist_ok=True
    )


    plt.figure(
        figsize=(6,5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "Actual Class"
    )


    plt.title(
        "Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        "../results/generated_figures/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# Main execution

if __name__ == "__main__":


    data = load_data()


    y_test, predictions = train_classifier(
        data
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    report = classification_report(
        y_test,
        predictions
    )


    print(
        "Accuracy:",
        accuracy
    )


    print(
        report
    )


    os.makedirs(
        "../results",
        exist_ok=True
    )


    with open(
        "../results/classification_report.txt",
        "w"
    ) as file:

        file.write(report)



    plot_confusion_matrix(
        y_test,
        predictions
    )


    print(
        "Classification analysis completed."
    )
