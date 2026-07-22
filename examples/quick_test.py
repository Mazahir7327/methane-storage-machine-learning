"""
Quick test for methane storage machine learning workflow.

This script verifies that the required Python environment
and machine learning libraries are correctly installed.
"""

import numpy as np
import pandas as pd
import sklearn
import xgboost


def quick_test():

    print("Methane storage ML workflow quick test")
    print("-------------------------------------")

    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("XGBoost:", xgboost.__version__)

    print("-------------------------------------")
    print("Environment successfully configured.")


if __name__ == "__main__":
    quick_test()
