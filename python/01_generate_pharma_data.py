import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------
# Reproducibility
# --------------------------------------------

np.random.seed(42)

N = 15000


# --------------------------------------------
# Generate batch variables
# --------------------------------------------

data = pd.DataFrame({

    "Batch_ID": [
        f"BATCH_{i:05d}"
        for i in range(1, N + 1)
    ],

    "Product": np.random.choice(
        ["Product_A", "Product_B", "Product_C"],
        N,
        p=[0.40, 0.35, 0.25]
    ),

    "Reactor": np.random.choice(
        ["Reactor_01", "Reactor_02",
         "Reactor_03", "Reactor_04"],
        N
    ),

    "Temperature_C": np.random.normal(
        27, 2.2, N
    ),

    "pH": np.random.normal(
        6.3, 0.25, N
    ),

    "Agitation_rpm": np.random.normal(
        275, 35, N
    ),

    "Process_Time_hr": np.random.normal(
        8.5, 1.0, N
    ),

    "Pressure_bar": np.random.normal(
        2.0, 0.30, N
    ),

    "Raw_Material_Purity_pct": np.random.normal(
        98.5, 0.8, N
    ),

    "Operator_Experience_yr": np.random.uniform(
        0.5, 15, N
    ),

    "Equipment_Age_yr": np.random.uniform(
        1, 12, N
    ),

    "Room_Humidity_pct": np.random.normal(
        50, 8, N
    ),

    "Intermediate_Impurity_pct": np.random.normal(
        2.0, 0.45, N
    ),

    "Shift": np.random.choice(
        ["Morning", "Evening", "Night"],
        N
    ),

    "Supplier": np.random.choice(
        ["Supplier_A", "Supplier_B", "Supplier_C"],
        N
    )
})


# --------------------------------------------
# Keep values within realistic ranges
# --------------------------------------------

data["Temperature_C"] = data["Temperature_C"].clip(20, 35)
data["pH"] = data["pH"].clip(5.3, 7.3)
data["Agitation_rpm"] = data["Agitation_rpm"].clip(150, 400)
data["Process_Time_hr"] = data["Process_Time_hr"].clip(5, 13)
data["Pressure_bar"] = data["Pressure_bar"].clip(1, 3)
data["Raw_Material_Purity_pct"] = (
    data["Raw_Material_Purity_pct"].clip(94, 100)
)
data["Room_Humidity_pct"] = (
    data["Room_Humidity_pct"].clip(25, 80)
)
data["Intermediate_Impurity_pct"] = (
    data["Intermediate_Impurity_pct"].clip(0.5, 4)
)


# --------------------------------------------
# Build hidden quality relationship
# --------------------------------------------

quality = np.full(N, 92.0)


# Temperature nonlinear penalty
quality -= (
    np.maximum(
        np.abs(data["Temperature_C"] - 26.5) - 1.0,
        0
    ) ** 2
) * 0.9


# pH nonlinear penalty
quality -= (
    np.maximum(
        np.abs(data["pH"] - 6.25) - 0.15,
        0
    ) ** 2
) * 12


# Impurity effect
quality -= (
    data["Intermediate_Impurity_pct"] - 1.5
) * 3.2


# Raw material effect
quality += (
    data["Raw_Material_Purity_pct"] - 98
) * 1.2


# Process-time penalty
quality -= (
    np.abs(data["Process_Time_hr"] - 8.5)
) * 0.5


# Equipment effect
quality -= np.where(
    data["Reactor"] == "Reactor_03",
    1.2,
    0
)


# Supplier effect
quality -= np.where(
    data["Supplier"] == "Supplier_C",
    0.8,
    0
)


# Interaction: high temperature + high impurity
interaction = (
    (data["Temperature_C"] > 29)
    &
    (data["Intermediate_Impurity_pct"] > 2.3)
)

quality -= np.where(
    interaction,
    4.0,
    0
)


# Random process variation
quality += np.random.normal(
    0,
    1.8,
    N
)


# Final target
data["Quality_Score"] = np.clip(
    quality,
    50,
    100
).round(2)


# --------------------------------------------
# Add limited missing data
# --------------------------------------------

for column in [
    "pH",
    "Raw_Material_Purity_pct",
    "Room_Humidity_pct"
]:

    missing_index = np.random.choice(
        data.index,
        size=int(0.01 * N),
        replace=False
    )

    data.loc[
        missing_index,
        column
    ] = np.nan


# --------------------------------------------
# Save dataset
# --------------------------------------------

project_root = Path(__file__).resolve().parent.parent

output_file = (
    project_root
    / "data"
    / "pharma_quality_data.csv"
)

data.to_csv(
    output_file,
    index=False
)


# --------------------------------------------
# Summary
# --------------------------------------------

print("\nDATASET CREATED")
print("-----------------------------")

print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nQuality Score Summary:")
print(data["Quality_Score"].describe())

print("\nMissing Values:")
print(data.isnull().sum())

print(
    "\nSaved to:",
    output_file
)