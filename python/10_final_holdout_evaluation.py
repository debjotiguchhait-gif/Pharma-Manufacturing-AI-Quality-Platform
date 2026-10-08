import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
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
# RECREATE ORIGINAL HOLDOUT SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDevelopment rows:", len(X_train))
print("Final holdout rows:", len(X_test))


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
# FINALISTS
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


# CV values obtained previously
cv_results = {

    "Gradient Boosting": {
        "CV_R2": 0.8762,
        "CV_MAE": 1.4752,
        "CV_RMSE": 1.8476
    },

    "MLP Neural Network": {
        "CV_R2": 0.8645,
        "CV_MAE": 1.5378,
        "CV_RMSE": 1.9328
    }
}


# ============================================
# FINAL HOLDOUT TEST
# ============================================

results = []


print("\nFINAL HOLDOUT EVALUATION")
print("=" * 75)


for model_name, pipeline in models.items():

    print(f"\nEvaluating: {model_name}")

    # Train using ALL development data
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict untouched holdout
    predictions = pipeline.predict(
        X_test
    )


    holdout_r2 = r2_score(
        y_test,
        predictions
    )

    holdout_mae = mean_absolute_error(
        y_test,
        predictions
    )

    holdout_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # Difference between CV and final holdout
    r2_gap = (
        cv_results[model_name]["CV_R2"]
        - holdout_r2
    )


    results.append({

        "Model": model_name,

        "CV_R2":
            cv_results[model_name]["CV_R2"],

        "Holdout_R2":
            holdout_r2,

        "R2_Generalization_Gap":
            r2_gap,

        "CV_MAE":
            cv_results[model_name]["CV_MAE"],

        "Holdout_MAE":
            holdout_mae,

        "CV_RMSE":
            cv_results[model_name]["CV_RMSE"],

        "Holdout_RMSE":
            holdout_rmse
    })


# ============================================
# RESULTS
# ============================================

results_df = pd.DataFrame(results)


numeric_columns = [
    "CV_R2",
    "Holdout_R2",
    "R2_Generalization_Gap",
    "CV_MAE",
    "Holdout_MAE",
    "CV_RMSE",
    "Holdout_RMSE"
]


results_df[numeric_columns] = (
    results_df[numeric_columns]
    .round(4)
)


results_df = results_df.sort_values(
    by=[
        "Holdout_R2",
        "Holdout_RMSE",
        "Holdout_MAE"
    ],
    ascending=[
        False,
        True,
        True
    ]
).reset_index(drop=True)


print("\n\nFINAL HOLDOUT RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================
# FINAL STATUS
# ============================================

results_df["Final_Status"] = "Finalist"

results_df.loc[
    0,
    "Final_Status"
] = "Champion"

results_df.loc[
    1,
    "Final_Status"
] = "Challenger"


champion = results_df.iloc[0]
challenger = results_df.iloc[1]


print("\n\nFINAL MODEL DECISION")
print("=" * 70)

print(
    "Champion:",
    champion["Model"]
)

print(
    "Challenger:",
    challenger["Model"]
)

print(
    "\nChampion Holdout R²:",
    champion["Holdout_R2"]
)

print(
    "Champion Holdout MAE:",
    champion["Holdout_MAE"]
)

print(
    "Champion Holdout RMSE:",
    champion["Holdout_RMSE"]
)

print(
    "Champion CV-to-Holdout R² Gap:",
    champion["R2_Generalization_Gap"]
)


# ============================================
# SAVE RESULTS
# ============================================

output_file = (
    project_root
    / "outputs"
    / "final_holdout_results.csv"
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
    "\nFINAL HOLDOUT EVALUATION COMPLETE."
)