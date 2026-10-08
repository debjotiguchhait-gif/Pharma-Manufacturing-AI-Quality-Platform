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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# ============================================
# LOAD DATA
# ============================================

project_root = Path(__file__).resolve().parent.parent

data_file = (
    project_root
    / "data"
    / "pharma_quality_data.csv"
)

df = pd.read_csv(data_file)


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
# HOLDOUT SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# IMPORTANT:
# Cross-validation will be performed ONLY on X_train/y_train.
# X_test/y_test remains untouched for later final evaluation.


# ============================================
# PREPROCESSING
# ============================================

numeric_scaled = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


numeric_unscaled = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


scaled_preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)


tree_preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)


# ============================================
# MODELS
# ============================================

models = {

    "Linear Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                scaled_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "Extra Trees": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=42
                )
            )
        ]
    ),

    "SVR": Pipeline(
        steps=[
            (
                "preprocessor",
                scaled_preprocessor
            ),
            (
                "model",
                SVR(
                    kernel="rbf"
                )
            )
        ]
    ),

    "MLP Neural Network": Pipeline(
        steps=[
            (
                "preprocessor",
                scaled_preprocessor
            ),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
            )
        ]
    )
}


# ============================================
# 5-FOLD CROSS-VALIDATION
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


results = []


print("\n5-FOLD CROSS-VALIDATION")
print("=" * 70)


for model_name, pipeline in models.items():

    print(f"\nEvaluating: {model_name}")

    start_time = time.time()

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    elapsed_time = time.time() - start_time


    cv_r2_mean = np.mean(
        scores["test_r2"]
    )

    cv_r2_std = np.std(
        scores["test_r2"]
    )

    cv_mae = -np.mean(
        scores["test_mae"]
    )

    cv_rmse = -np.mean(
        scores["test_rmse"]
    )


    results.append({

        "Model": model_name,

        "CV_R2_Mean": cv_r2_mean,

        "CV_R2_Std": cv_r2_std,

        "CV_MAE": cv_mae,

        "CV_RMSE": cv_rmse,

        "CV_Time_sec": elapsed_time

    })


# ============================================
# RESULTS
# ============================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="CV_R2_Mean",
    ascending=False
)


numeric_columns = [
    "CV_R2_Mean",
    "CV_R2_Std",
    "CV_MAE",
    "CV_RMSE",
    "CV_Time_sec"
]


results_df[numeric_columns] = (
    results_df[numeric_columns]
    .round(4)
)


print("\n\nCROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================
# INITIAL CV WINNER
# ============================================

best = results_df.iloc[0]


print("\n\nBEST CROSS-VALIDATED MODEL")
print("=" * 70)

print("Model:", best["Model"])
print("Mean CV R²:", best["CV_R2_Mean"])
print("CV R² SD:", best["CV_R2_Std"])
print("Mean CV MAE:", best["CV_MAE"])
print("Mean CV RMSE:", best["CV_RMSE"])


# ============================================
# SAVE RESULTS
# ============================================

output_file = (
    project_root
    / "outputs"
    / "cross_validation_results.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:",
    output_file
)