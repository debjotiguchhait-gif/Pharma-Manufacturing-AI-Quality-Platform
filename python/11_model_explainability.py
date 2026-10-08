import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.inspection import permutation_importance

from sklearn.metrics import r2_score


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


numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


# ============================================
# RECREATE DEVELOPMENT / HOLDOUT SPLIT
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
# FINAL CHAMPION MODEL
# ============================================

champion = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=10,
            min_samples_leaf=1,
            subsample=0.85,
            random_state=42
        )
    )
])


# ============================================
# TRAIN CHAMPION
# ============================================

print("\nTraining frozen Champion model...")

champion.fit(
    X_train,
    y_train
)


predictions = champion.predict(
    X_test
)

holdout_r2 = r2_score(
    y_test,
    predictions
)


print(
    "Champion Holdout R²:",
    round(holdout_r2, 4)
)


# ============================================
# PERMUTATION IMPORTANCE
# ============================================

print("\nCalculating permutation importance...")


importance = permutation_importance(
    champion,
    X_test,
    y_test,

    scoring="r2",

    # Repeat each permutation for stability
    n_repeats=10,

    random_state=42,

    n_jobs=-1
)


# ============================================
# BUILD IMPORTANCE TABLE
# ============================================

importance_df = pd.DataFrame({

    "Feature": X_test.columns,

    "Importance_Mean":
        importance.importances_mean,

    "Importance_Std":
        importance.importances_std

})


importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
).reset_index(drop=True)


importance_df["Importance_Mean"] = (
    importance_df["Importance_Mean"]
    .round(4)
)

importance_df["Importance_Std"] = (
    importance_df["Importance_Std"]
    .round(4)
)


# ============================================
# DISPLAY RESULTS
# ============================================

print("\nGLOBAL FEATURE IMPORTANCE")
print("=" * 65)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================
# TOP PROCESS DRIVERS
# ============================================

print("\nTOP 5 MODEL DRIVERS")
print("=" * 65)

print(
    importance_df
    .head(5)
    .to_string(index=False)
)


# ============================================
# SAVE
# ============================================

output_file = (
    project_root
    / "outputs"
    / "global_feature_importance.csv"
)


importance_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:",
    output_file
)