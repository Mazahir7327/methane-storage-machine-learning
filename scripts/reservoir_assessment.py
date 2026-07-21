"""
Reservoir Assessment Visualization
----------------------------------

This script generates reservoir characterization figures
for saline reservoirs and depleted gas reservoirs.

Used for:
- Porosity vs permeability analysis
- Maximum gas saturation analysis
- Caprock thickness distribution
- Composite storage suitability visualization

Author:
Mazahir Hussain

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==========================================================
# File locations
# ==========================================================

DATA_FOLDER = "../data"

saline_file = os.path.join(
    DATA_FOLDER,
    "saline_reservoir_example.csv"
)

dgr_file = os.path.join(
    DATA_FOLDER,
    "depleted_gas_reservoir_example.csv"
)


# ==========================================================
# Column definitions
# ==========================================================

porosity_col = "Porosity"
permeability_col = "Permeability"

gas_saturation_col = "Maximum Gas Saturation"

caprock_col = "Caprock Thickness"

pressure_col = "Pressure Coefficient"

lithology_col = "Lithology"



# ==========================================================
# Reservoir plotting function
# ==========================================================

def plot_reservoir_assessment(df, title, output_name):

    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18,12)
    )

    # --------------------------
    # Porosity vs permeability
    # --------------------------

    sns.scatterplot(
        data=df,
        x=porosity_col,
        y=permeability_col,
        hue=lithology_col,
        s=80,
        ax=axes[0,0]
    )

    axes[0,0].set_title(
        "(A) Porosity vs Permeability"
    )


    # --------------------------
    # Gas saturation vs permeability
    # --------------------------

    sns.scatterplot(
        data=df,
        x=gas_saturation_col,
        y=permeability_col,
        hue=lithology_col,
        s=80,
        ax=axes[0,1]
    )

    axes[0,1].set_title(
        "(B) Gas Saturation vs Permeability"
    )


    # --------------------------
    # Porosity vs gas saturation
    # --------------------------

    sns.scatterplot(
        data=df,
        x=porosity_col,
        y=gas_saturation_col,
        hue=lithology_col,
        s=80,
        ax=axes[0,2]
    )

    axes[0,2].set_title(
        "(C) Porosity vs Gas Saturation"
    )


    # --------------------------
    # Caprock thickness
    # --------------------------

    sns.boxplot(
        data=df,
        x=lithology_col,
        y=caprock_col,
        ax=axes[1,0]
    )

    axes[1,0].set_title(
        "(D) Caprock Thickness"
    )


    # --------------------------
    # Pressure relationship
    # --------------------------

    sns.scatterplot(
        data=df,
        x=pressure_col,
        y=gas_saturation_col,
        hue=lithology_col,
        s=80,
        ax=axes[1,1]
    )

    axes[1,1].set_title(
        "(E) Pressure vs Gas Saturation"
    )


    # --------------------------
    # Composite suitability
    # --------------------------

    scatter = axes[1,2].scatter(
        df[porosity_col],
        df[gas_saturation_col],
        s=df[caprock_col]*10,
        c=df[permeability_col],
        cmap="viridis",
        alpha=0.7
    )


    axes[1,2].set_title(
        "(F) Composite Suitability"
    )


    axes[1,2].set_xlabel(
        "Porosity (%)"
    )

    axes[1,2].set_ylabel(
        "Maximum Gas Saturation (%)"
    )


    plt.colorbar(
        scatter,
        ax=axes[1,2],
        label="Permeability"
    )


    plt.suptitle(
        title,
        fontsize=18
    )


    plt.tight_layout()


    os.makedirs(
        "../results/generated_figures",
        exist_ok=True
    )


    plt.savefig(
        "../results/generated_figures/" + output_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ==========================================================
# Main execution
# ==========================================================


if __name__ == "__main__":


    saline_data = pd.read_csv(
        saline_file
    )


    dgr_data = pd.read_csv(
        dgr_file
    )


    plot_reservoir_assessment(
        saline_data,
        "Saline Reservoir Assessment",
        "Figure6_saline_reservoir.png"
    )


    plot_reservoir_assessment(
        dgr_data,
        "Depleted Gas Reservoir Assessment",
        "Figure7_depleted_reservoir.png"
    )


    print(
        "Reservoir assessment figures generated successfully."
    )
