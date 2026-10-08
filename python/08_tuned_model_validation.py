import pandas as pd
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.neural_network import MLPRegressor


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
# DEVELOPMENT / HOLDOUT SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDevelopment rows:", len(X_train))
print("Untouched holdout rows:", len(X_test))


# ============================================
# PREPROCESSING
# ============================================

numeric_scaled = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


numeric_unscaled = Pipeline([
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


scaled_preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_scaled,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])


tree_preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_unscaled,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])


# ============================================
# TUNED MODELS
# Exact parameters from Step 7
# ============================================

models = {

    "Gradient Boosting": Pipeline([
        (
            "preprocessor",
            tree_preprocessor
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
    ]),


    "Random Forest": Pipeline([
        (
            "preprocessor",
            tree_preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features=0.75,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    "Extra Trees": Pipeline([
        (
            "preprocessor",
            tree_preprocessor
        ),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features=0.75,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    "MLP Neural Network": Pipeline([
        (
            "preprocessor",
            scaled_preprocessor
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(128,),
                activation="tanh",
                alpha=0.001,
                learning_rate_init=0.005,
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        )
    ])
}


# ============================================
# 5-FOLD CV
# ============================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error"
}


# ============================================
# VALIDATION
# ============================================

results = []

print("\nTUNED MODEL VALIDATION")
print("=" * 75)


for model_name, pipeline in models.items():

    print(f"\nValidating: {model_name}")

    start = time.time()

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    elapsed = time.time() - start

    results.append({

        "Model": model_name,

        "CV_R2_Mean":
            np.mean(scores["test_r2"]),

        "CV_R2_Std":
            np.std(scores["test_r2"]),

        "CV_MAE":
            -np.mean(scores["test_mae"]),

        "CV_RMSE":
            -np.mean(scores["test_rmse"]),

        "Validation_Time_sec":
            elapsed
    })


# ============================================
# RESULTS
# ============================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_R2_Mean",
    ascending=False
)

numeric_cols = [
    "CV_R2_Mean",
    "CV_R2_Std",
    "CV_MAE",
    "CV_RMSE",
    "Validation_Time_sec"
]

results_df[numeric_cols] = (
    results_df[numeric_cols]
    .round(4)
)


print("\n\nTUNED MODEL VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================
# SAVE
# ============================================

output_file = (
    project_root
    / "outputs"
    / "tuned_model_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:",
    output_file
)

print(
    "\nHoldout test set remains untouched."
)