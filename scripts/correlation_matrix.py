"""
Correlation Matrix Analysis for Methane Storage Assessment

This script calculates Pearson correlation coefficients
between reservoir parameters and generates a heatmap.

Author:
Mazahir Hussain
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset path
DATA_FILE = "../data/methane_storage_dataset.csv"


def load_data():

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip()

    return df



def correlation_analysis(df):

    correlation = df.corr(
        method="pearson"
    )

    return correlation



def plot_correlation(correlation):

    os.makedirs(
        "../results/generated_figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(10,8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Matrix of Reservoir Parameters"
    )

    plt.tight_layout()


    plt.savefig(
        "../results/generated_figures/correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



if __name__ == "__main__":


    data = load_data()

    correlation = correlation_analysis(
        data
    )


    print(correlation)


    os.makedirs(
        "../results",
        exist_ok=True
    )


    correlation.to_csv(
        "../results/correlation_matrix.csv"
    )


    plot_correlation(
        correlation
    )


    print(
        "Correlation analysis completed."
    )
