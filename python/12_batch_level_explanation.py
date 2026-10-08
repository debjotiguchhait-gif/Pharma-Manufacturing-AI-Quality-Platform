import pandas as pd
import numpy as np
import shap
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


# ============================================
# LOAD DATA
# ============================================

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root
    / "data"
    / "pharma_quality_data.csv"
)


# ============================================
# FEATURES / TARGET
# ============================================

TARGET = "Quality_Score"

X = df.drop(
    columns=[TARGET, "Batch_ID"]
)

y = df[TARGET]

batch_ids = df["Batch_ID"]


numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


# ============================================
# RECREATE ORIGINAL SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================
# PREPROCESSING
# ============================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])


# ============================================
# CHAMPION MODEL
# ============================================

model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=1,
    subsample=0.85,
    random_state=42
)


champion = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================
# TRAIN CHAMPION
# ============================================

print("\nTraining Champion model...")

champion.fit(
    X_train,
    y_train
)


# ============================================
# SELECT EXAMPLE HOLDOUT BATCH
# ============================================

# Select the holdout batch with the lowest
# actual Quality Score for demonstration.

example_index = y_test.idxmin()

batch = X.loc[[example_index]]

actual_quality = y.loc[example_index]

batch_id = batch_ids.loc[example_index]


prediction = champion.predict(
    batch
)[0]


print("\nSELECTED BATCH")
print("=" * 70)

print("Batch ID:", batch_id)

print(
    "Actual Quality Score:",
    round(actual_quality, 2)
)

print(
    "Predicted Quality Score:",
    round(prediction, 2)
)


# ============================================
# TRANSFORM BATCH
# ============================================

fitted_preprocessor = (
    champion.named_steps["preprocessor"]
)

fitted_model = (
    champion.named_steps["model"]
)


batch_transformed = (
    fitted_preprocessor.transform(batch)
)


# ============================================
# GET TRANSFORMED FEATURE NAMES
# ============================================

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

# Clean sklearn prefixes for readability

feature_names = [
    name
    .replace("numeric__", "")
    .replace("categorical__", "")
    for name in feature_names
]


# ============================================
# SHAP EXPLAINER
# ============================================

explainer = shap.TreeExplainer(
    fitted_model
)

shap_values = explainer.shap_values(
    batch_transformed
)


# Handle output shape safely
shap_array = np.array(
    shap_values
).reshape(-1)


# ============================================
# BUILD CONTRIBUTION TABLE
# ============================================

feature_values = (
    np.array(batch_transformed)
    .reshape(-1)
)


explanation_df = pd.DataFrame({

    "Feature": feature_names,

    "Feature_Value": feature_values,

    "SHAP_Value": shap_array

})


explanation_df[
    "Absolute_SHAP"
] = np.abs(
    explanation_df["SHAP_Value"]
)


explanation_df = (
    explanation_df
    .sort_values(
        by="Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================
# DISPLAY TOP CONTRIBUTIONS
# ============================================

print("\nTOP BATCH-LEVEL MODEL CONTRIBUTIONS")
print("=" * 80)

print(
    explanation_df[
        [
            "Feature",
            "Feature_Value",
            "SHAP_Value"
        ]
    ]
    .head(10)
    .round(4)
    .to_string(index=False)
)


# ============================================
# DIRECTION
# ============================================

explanation_df["Direction"] = np.where(
    explanation_df["SHAP_Value"] > 0,
    "Increases prediction",
    "Decreases prediction"
)


print("\nTOP 5 EXPLANATIONS")
print("=" * 80)

print(
    explanation_df[
        [
            "Feature",
            "Feature_Value",
            "SHAP_Value",
            "Direction"
        ]
    ]
    .head(5)
    .round(4)
    .to_string(index=False)
)


# ============================================
# SHAP RECONSTRUCTION CHECK
# ============================================

base_value = float(
    np.array(
        explainer.expected_value
    ).reshape(-1)[0]
)

shap_sum = (
    base_value
    + shap_array.sum()
)


print("\nSHAP RECONSTRUCTION")
print("=" * 70)

print(
    "Baseline prediction:",
    round(base_value, 4)
)

print(
    "Sum of SHAP contributions:",
    round(shap_array.sum(), 4)
)

print(
    "Reconstructed prediction:",
    round(shap_sum, 4)
)

print(
    "Model prediction:",
    round(prediction, 4)
)


# ============================================
# SAVE
# ============================================

output_file = (
    project_root
    / "outputs"
    / "batch_level_explanation.csv"
)


explanation_df.to_csv(
    output_file,
    index=False
)


print(
    "\nExplanation saved to:",
    output_file
)