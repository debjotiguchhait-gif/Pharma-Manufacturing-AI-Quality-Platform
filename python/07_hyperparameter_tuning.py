import pandas as pd
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

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
# DEVELOPMENT / HOLDOUT SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDEVELOPMENT ROWS:", len(X_train))
print("UNTOUCHED HOLDOUT ROWS:", len(X_test))


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
            SimpleImputer(strategy="most_frequent")
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
# MODEL PIPELINES
# ============================================

models = {

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

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
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
                    random_state=42,
                    n_jobs=-1
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
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
            )
        ]
    )
}


# ============================================
# PARAMETER SEARCH SPACES
# ============================================

parameter_spaces = {

    "Gradient Boosting": {

        "model__n_estimators": [
            100, 200, 300, 400
        ],

        "model__learning_rate": [
            0.03, 0.05, 0.1, 0.15
        ],

        "model__max_depth": [
            2, 3, 4, 5
        ],

        "model__min_samples_split": [
            2, 5, 10
        ],

        "model__min_samples_leaf": [
            1, 2, 4
        ],

        "model__subsample": [
            0.7, 0.85, 1.0
        ]
    },


    "Random Forest": {

        "model__n_estimators": [
            200, 300, 500
        ],

        "model__max_depth": [
            None, 10, 20, 30
        ],

        "model__min_samples_split": [
            2, 5, 10
        ],

        "model__min_samples_leaf": [
            1, 2, 4
        ],

        "model__max_features": [
            0.5, 0.75, 1.0
        ]
    },


    "Extra Trees": {

        "model__n_estimators": [
            200, 300, 500
        ],

        "model__max_depth": [
            None, 10, 20, 30
        ],

        "model__min_samples_split": [
            2, 5, 10
        ],

        "model__min_samples_leaf": [
            1, 2, 4
        ],

        "model__max_features": [
            0.5, 0.75, 1.0
        ]
    },


    "MLP Neural Network": {

        "model__hidden_layer_sizes": [
            (64,),
            (128,),
            (64, 32),
            (128, 64),
            (128, 64, 32)
        ],

        "model__activation": [
            "relu",
            "tanh"
        ],

        "model__alpha": [
            0.0001,
            0.001,
            0.01
        ],

        "model__learning_rate_init": [
            0.0005,
            0.001,
            0.005
        ]
    }
}


# ============================================
# CROSS-VALIDATION
# ============================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================
# RANDOMIZED SEARCH
# ============================================

results = []
best_estimators = {}


print("\nHYPERPARAMETER TUNING")
print("=" * 70)


for model_name, pipeline in models.items():

    print(f"\nTuning: {model_name}")

    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_spaces[model_name],

        # 15 random parameter combinations
        n_iter=15,

        scoring="r2",

        cv=cv,

        random_state=42,

        # Keep at 1 for predictable local execution
        n_jobs=1,

        verbose=1,

        refit=True
    )


    search.fit(
        X_train,
        y_train
    )


    elapsed_time = (
        time.time() - start_time
    )


    best_estimators[
        model_name
    ] = search.best_estimator_


    results.append({

        "Model": model_name,

        "Best_CV_R2":
            search.best_score_,

        "Tuning_Time_sec":
            elapsed_time,

        "Best_Parameters":
            search.best_params_

    })


    print(
        "Best CV R²:",
        round(search.best_score_, 4)
    )

    print(
        "Best Parameters:",
        search.best_params_
    )


# ============================================
# RESULTS
# ============================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="Best_CV_R2",
    ascending=False
)


results_df["Best_CV_R2"] = (
    results_df["Best_CV_R2"]
    .round(4)
)

results_df["Tuning_Time_sec"] = (
    results_df["Tuning_Time_sec"]
    .round(2)
)


print("\n\nTUNING RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "Model",
            "Best_CV_R2",
            "Tuning_Time_sec"
        ]
    ].to_string(index=False)
)


# ============================================
# BEST TUNED CANDIDATE
# ============================================

best_candidate = (
    results_df.iloc[0]
)


print("\n\nBEST TUNED CANDIDATE")
print("=" * 70)

print(
    "Model:",
    best_candidate["Model"]
)

print(
    "CV R²:",
    best_candidate["Best_CV_R2"]
)

print(
    "Parameters:"
)

print(
    best_candidate["Best_Parameters"]
)


# ============================================
# SAVE RESULTS
# ============================================

output_file = (
    project_root
    / "outputs"
    / "hyperparameter_tuning_results.csv"
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
    "\nNOTE: Holdout test data has NOT been "
    "used for model selection."
)