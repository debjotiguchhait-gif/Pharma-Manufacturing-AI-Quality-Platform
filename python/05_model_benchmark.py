import pandas as pd
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


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
# TRAIN / TEST SPLIT
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

# Used for models requiring scaling
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


# Used for tree models
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
# MODEL DEFINITIONS
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
# MODEL BENCHMARK
# ============================================

results = []


print("\nMODEL BENCHMARK")
print("=" * 70)


for model_name, pipeline in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    train_prediction = pipeline.predict(
        X_train
    )

    test_prediction = pipeline.predict(
        X_test
    )


    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    generalization_gap = (
        train_r2 - test_r2
    )


    results.append({

        "Model": model_name,

        "Train_R2": train_r2,

        "Test_R2": test_r2,

        "R2_Gap": generalization_gap,

        "MAE": mae,

        "RMSE": rmse,

        "Training_Time_sec": training_time

    })


# ============================================
# RESULTS TABLE
# ============================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="Test_R2",
    ascending=False
)


numeric_columns = [
    "Train_R2",
    "Test_R2",
    "R2_Gap",
    "MAE",
    "RMSE",
    "Training_Time_sec"
]

results_df[numeric_columns] = (
    results_df[numeric_columns]
    .round(4)
)


print("\n\nMODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================
# INITIAL BEST MODEL
# ============================================

best_model = results_df.iloc[0]


print("\n\nINITIAL BEST MODEL")
print("=" * 70)

print(
    "Model:",
    best_model["Model"]
)

print(
    "Test R²:",
    best_model["Test_R2"]
)

print(
    "MAE:",
    best_model["MAE"]
)

print(
    "RMSE:",
    best_model["RMSE"]
)

print(
    "Train-Test R² Gap:",
    best_model["R2_Gap"]
)


# ============================================
# SAVE BENCHMARK RESULTS
# ============================================

output_file = (
    project_root
    / "outputs"
    / "model_benchmark.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:",
    output_file
)