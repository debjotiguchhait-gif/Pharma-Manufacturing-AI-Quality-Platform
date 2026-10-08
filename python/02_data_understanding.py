import pandas as pd
from pathlib import Path


# --------------------------------------------
# Load data
# --------------------------------------------

project_root = Path(__file__).resolve().parent.parent

data_file = (
    project_root
    / "data"
    / "pharma_quality_data.csv"
)

df = pd.read_csv(data_file)


# --------------------------------------------
# Basic information
# --------------------------------------------

print("\nDATASET OVERVIEW")
print("-----------------------------")

print("Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)


# --------------------------------------------
# Target analysis
# --------------------------------------------

print("\nQUALITY SCORE")
print("-----------------------------")

print(df["Quality_Score"].describe())


# --------------------------------------------
# Missing data
# --------------------------------------------

print("\nMISSING VALUES")
print("-----------------------------")

missing = df.isnull().sum()

print(
    missing[missing > 0]
)


# --------------------------------------------
# Numerical correlations with target
# --------------------------------------------

print("\nCORRELATION WITH QUALITY SCORE")
print("-----------------------------")

numeric_df = df.select_dtypes(
    include="number"
)

correlation = (
    numeric_df
    .corr()["Quality_Score"]
    .sort_values(ascending=False)
)

print(correlation)


# --------------------------------------------
# Product performance
# --------------------------------------------

print("\nQUALITY BY PRODUCT")
print("-----------------------------")

print(
    df.groupby("Product")["Quality_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)


# --------------------------------------------
# Reactor performance
# --------------------------------------------

print("\nQUALITY BY REACTOR")
print("-----------------------------")

print(
    df.groupby("Reactor")["Quality_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)


# --------------------------------------------
# Supplier performance
# --------------------------------------------

print("\nQUALITY BY SUPPLIER")
print("-----------------------------")

print(
    df.groupby("Supplier")["Quality_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)


# --------------------------------------------
# Shift performance
# --------------------------------------------

print("\nQUALITY BY SHIFT")
print("-----------------------------")

print(
    df.groupby("Shift")["Quality_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)