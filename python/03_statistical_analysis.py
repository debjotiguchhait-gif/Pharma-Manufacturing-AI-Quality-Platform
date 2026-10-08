import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, f_oneway


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


# ============================================
# CONTINUOUS VARIABLES
# ============================================

print("\nCONTINUOUS VARIABLE ANALYSIS")
print("=" * 45)

continuous_variables = [
    "Temperature_C",
    "Intermediate_Impurity_pct",
    "Raw_Material_Purity_pct",
    "pH"
]

for variable in continuous_variables:

    temp = df[
        [variable, "Quality_Score"]
    ].dropna()

    correlation, p_value = pearsonr(
        temp[variable],
        temp["Quality_Score"]
    )

    print(f"\n{variable}")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")


# ============================================
# CATEGORICAL VARIABLES
# ============================================

print("\n\nCATEGORICAL VARIABLE ANALYSIS")
print("=" * 45)

categorical_variables = [
    "Product",
    "Reactor",
    "Supplier",
    "Shift"
]

for variable in categorical_variables:

    groups = [
        group["Quality_Score"].values
        for _, group in df.groupby(variable)
    ]

    f_stat, p_value = f_oneway(*groups)

    print(f"\n{variable}")
    print(f"ANOVA F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4e}")


# ============================================
# GROUP MEANS FOR SIGNIFICANT CATEGORICAL DATA
# ============================================

print("\n\nCATEGORY MEANS")
print("=" * 45)

for variable in categorical_variables:

    print(f"\n{variable}")

    means = (
        df.groupby(variable)["Quality_Score"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    print(means)