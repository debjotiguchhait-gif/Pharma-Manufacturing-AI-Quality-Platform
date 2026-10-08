import pandas as pd
from pathlib import Path


# ============================================
# PATHS
# ============================================

project_root = Path(__file__).resolve().parent.parent

validation_file = (
    project_root
    / "outputs"
    / "tuned_model_validation.csv"
)

output_file = (
    project_root
    / "outputs"
    / "model_selection_results.csv"
)


# ============================================
# LOAD VALIDATED MODEL RESULTS
# ============================================

df = pd.read_csv(validation_file)

print("\nVALIDATED MODELS")
print("=" * 70)

print(df.to_string(index=False))


# ============================================
# ELIGIBILITY GATES
# ============================================

# Minimum acceptable predictive performance
MIN_CV_R2 = 0.80

# Maximum acceptable fold-to-fold instability
MAX_CV_R2_STD = 0.02


df["Performance_Pass"] = (
    df["CV_R2_Mean"] >= MIN_CV_R2
)

df["Stability_Pass"] = (
    df["CV_R2_Std"] <= MAX_CV_R2_STD
)

df["Eligible"] = (
    df["Performance_Pass"]
    &
    df["Stability_Pass"]
)


# ============================================
# ELIGIBLE MODELS
# ============================================

eligible = df[
    df["Eligible"]
].copy()


if len(eligible) < 2:

    raise ValueError(
        "Fewer than two models passed the "
        "selection gates."
    )


# ============================================
# MULTI-METRIC RANKING
# ============================================

# Higher R² is better
eligible["R2_Rank"] = (
    eligible["CV_R2_Mean"]
    .rank(
        ascending=False,
        method="min"
    )
)

# Lower RMSE is better
eligible["RMSE_Rank"] = (
    eligible["CV_RMSE"]
    .rank(
        ascending=True,
        method="min"
    )
)

# Lower MAE is better
eligible["MAE_Rank"] = (
    eligible["CV_MAE"]
    .rank(
        ascending=True,
        method="min"
    )
)

# Lower variability is better
eligible["Stability_Rank"] = (
    eligible["CV_R2_Std"]
    .rank(
        ascending=True,
        method="min"
    )
)


# ============================================
# PRIMARY SELECTION POLICY
# ============================================

# We do NOT create arbitrary weighted scores.
#
# Selection hierarchy:
# 1. Highest CV R²
# 2. Lowest RMSE
# 3. Lowest MAE
# 4. Lowest CV variability

eligible = eligible.sort_values(
    by=[
        "CV_R2_Mean",
        "CV_RMSE",
        "CV_MAE",
        "CV_R2_Std"
    ],
    ascending=[
        False,
        True,
        True,
        True
    ]
).reset_index(drop=True)


# ============================================
# ASSIGN MODEL STATUS
# ============================================

eligible["Selection_Status"] = "Candidate"

eligible.loc[
    0,
    "Selection_Status"
] = "Champion Candidate"

eligible.loc[
    1,
    "Selection_Status"
] = "Challenger Candidate"


# ============================================
# RESULTS
# ============================================

columns = [
    "Model",
    "CV_R2_Mean",
    "CV_R2_Std",
    "CV_MAE",
    "CV_RMSE",
    "R2_Rank",
    "RMSE_Rank",
    "MAE_Rank",
    "Stability_Rank",
    "Selection_Status"
]


print("\n\nMODEL SELECTION RESULTS")
print("=" * 90)

print(
    eligible[columns]
    .to_string(index=False)
)


# ============================================
# CHAMPION / CHALLENGER
# ============================================

champion = eligible.iloc[0]
challenger = eligible.iloc[1]


print("\n\nSELECTION DECISION")
print("=" * 70)

print(
    "Champion Candidate:",
    champion["Model"]
)

print(
    "Challenger Candidate:",
    challenger["Model"]
)


print("\nChampion CV R²:", champion["CV_R2_Mean"])
print("Champion CV RMSE:", champion["CV_RMSE"])
print("Champion CV MAE:", champion["CV_MAE"])
print("Champion CV SD:", champion["CV_R2_Std"])


# ============================================
# SAVE
# ============================================

eligible.to_csv(
    output_file,
    index=False
)

print(
    "\nResults saved to:",
    output_file
)