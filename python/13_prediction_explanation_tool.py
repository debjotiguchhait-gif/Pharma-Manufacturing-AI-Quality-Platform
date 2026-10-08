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

feature_columns = [
    column
    for column in df.columns
    if column not in [TARGET, "Batch_ID"]
]

X = df[feature_columns]

y = df[TARGET]


numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


# ============================================
# DEVELOPMENT DATA
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
# FROZEN CHAMPION MODEL
# ============================================

champion_model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=1,
    subsample=0.85,
    random_state=42
)


champion_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        champion_model
    )
])


# ============================================
# TRAIN
# ============================================

print("Loading Champion model...")

champion_pipeline.fit(
    X_train,
    y_train
)


# ============================================
# PREPARE SHAP
# ============================================

fitted_preprocessor = (
    champion_pipeline.named_steps[
        "preprocessor"
    ]
)

fitted_model = (
    champion_pipeline.named_steps[
        "model"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


feature_names = [
    name
    .replace("numeric__", "")
    .replace("categorical__", "")
    for name in feature_names
]


explainer = shap.TreeExplainer(
    fitted_model
)


# ============================================
# RISK CLASSIFICATION
# ============================================

def classify_risk(predicted_quality):

    """
    Demonstration-level risk bands.

    These are NOT validated pharmaceutical
    release specifications.
    """

    if predicted_quality < 75:
        return "HIGH"

    elif predicted_quality < 85:
        return "MODERATE"

    else:
        return "LOW"


# ============================================
# CLEAN FEATURE NAME
# ============================================

def clean_feature_name(feature):

    replacements = {

        "Temperature_C":
            "Temperature",

        "Intermediate_Impurity_pct":
            "Intermediate Impurity",

        "Raw_Material_Purity_pct":
            "Raw Material Purity",

        "Process_Time_hr":
            "Process Time",

        "Operator_Experience_yr":
            "Operator Experience",

        "Equipment_Age_yr":
            "Equipment Age",

        "Room_Humidity_pct":
            "Room Humidity",

        "Agitation_rpm":
            "Agitation",

        "Pressure_bar":
            "Pressure",

        "pH":
            "pH"
    }

    return replacements.get(
        feature,
        feature.replace("_", " ")
    )


# ============================================
# EXPLANATION FUNCTION
# ============================================

def explain_batch(
    batch_data,
    batch_id="NEW_BATCH",
    top_n=5
):

    """
    Predict quality and generate local
    SHAP explanation for one batch.

    Parameters
    ----------
    batch_data : dict
        Original process variables.

    batch_id : str
        Batch identifier.

    top_n : int
        Number of strongest contributions.

    Returns
    -------
    dict
        Structured prediction evidence.
    """

    # ----------------------------------------
    # Convert input to dataframe
    # ----------------------------------------

    batch_df = pd.DataFrame(
        [batch_data]
    )

    # Ensure correct feature order
    batch_df = batch_df[
        feature_columns
    ]


    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    prediction = float(
        champion_pipeline.predict(
            batch_df
        )[0]
    )


    # ----------------------------------------
    # Transform batch
    # ----------------------------------------

    transformed = (
        fitted_preprocessor
        .transform(batch_df)
    )


    transformed_array = np.array(
        transformed
    ).reshape(-1)


    # ----------------------------------------
    # SHAP
    # ----------------------------------------

    shap_values = (
        explainer.shap_values(
            transformed
        )
    )


    shap_array = np.array(
        shap_values
    ).reshape(-1)


    base_value = float(
        np.array(
            explainer.expected_value
        ).reshape(-1)[0]
    )


    # ----------------------------------------
    # Build explanation table
    # ----------------------------------------

    explanation_df = pd.DataFrame({

        "feature":
            feature_names,

        "feature_value":
            transformed_array,

        "shap_value":
            shap_array
    })


    explanation_df[
        "absolute_shap"
    ] = np.abs(
        explanation_df[
            "shap_value"
        ]
    )


    explanation_df = (
        explanation_df
        .sort_values(
            by="absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # ----------------------------------------
    # Human-readable SHAP drivers
    # ----------------------------------------

    drivers = []

    # Numerical features
    for feature in numeric_features:

        matching_row = explanation_df[
            explanation_df["feature"] == feature
        ]

        if matching_row.empty:
            continue

        row = matching_row.iloc[0]

        shap_value = float(
            row["shap_value"]
        )

        original_value = batch_data.get(
            feature
        )

        drivers.append({

            "feature":
                clean_feature_name(feature),

            "feature_type":
                "numeric",

            "observed_value":
                (
                    None
                    if pd.isna(original_value)
                    else round(
                        float(original_value),
                        4
                    )
                ),

            "shap_value":
                round(
                    shap_value,
                    4
                ),

            "direction":
                (
                    "increases prediction"
                    if shap_value > 0
                    else "decreases prediction"
                )
        })


    # ----------------------------------------
    # Categorical features
    # Aggregate one-hot SHAP values back to
    # the original feature level
    # ----------------------------------------

    for feature in categorical_features:

        prefix = f"{feature}_"

        matching_rows = explanation_df[
            explanation_df[
                "feature"
            ].str.startswith(prefix)
        ]

        if matching_rows.empty:
            continue


        total_shap = float(
            matching_rows[
                "shap_value"
            ].sum()
        )


        drivers.append({

            "feature":
                clean_feature_name(feature),

            "feature_type":
                "categorical",

            "observed_value":
                str(
                    batch_data.get(feature)
                ),

            "shap_value":
                round(
                    total_shap,
                    4
                ),

            "direction":
                (
                    "increases prediction"
                    if total_shap > 0
                    else "decreases prediction"
                )
        })


    # ----------------------------------------
    # Rank by absolute SHAP contribution
    # ----------------------------------------

    drivers = sorted(
        drivers,
        key=lambda x: abs(
            x["shap_value"]
        ),
        reverse=True
    )


    drivers = drivers[:top_n]

    # ----------------------------------------
    # Reconstruction check
    # ----------------------------------------

    reconstructed_prediction = (
        base_value
        + shap_array.sum()
    )


    # ----------------------------------------
    # Structured evidence package
    # ----------------------------------------

    result = {

        "batch_id":
            batch_id,

        "prediction": {

            "quality_score":
                round(prediction, 2),

            "risk_level":
                classify_risk(
                    prediction
                )
        },

        "explanation": {

            "baseline_prediction":
                round(
                    base_value,
                    4
                ),

            "top_model_drivers":
                drivers,

            "shap_reconstructed_prediction":
                round(
                    float(
                        reconstructed_prediction
                    ),
                    4
                )
        },

        "model_information": {

            "model":
                "Gradient Boosting",

            "holdout_r2":
                0.8772,

            "holdout_mae":
                1.4314,

            "holdout_rmse":
                1.8205
        },

        "guardrails": {

            "causality_warning":
                (
                    "SHAP values explain model "
                    "predictions and do not "
                    "establish causation."
                ),

            "risk_warning":
                (
                    "Risk categories are "
                    "demonstration-level bands "
                    "and are not validated "
                    "pharmaceutical release "
                    "specifications."
                )
        }
    }


    return result


# ============================================
# TEST WITH BATCH_11325
# ============================================

if __name__ == "__main__":

    example_index = (
        df[
            df["Batch_ID"]
            == "BATCH_11325"
        ].index[0]
    )


    example_batch = (
        df.loc[
            example_index,
            feature_columns
        ]
        .to_dict()
    )


    result = explain_batch(
        batch_data=example_batch,
        batch_id="BATCH_11325",
        top_n=5
    )


    print("\nPREDICTION EVIDENCE")
    print("=" * 70)

    print(
        "\nBatch:",
        result["batch_id"]
    )

    print(
        "Predicted Quality:",
        result[
            "prediction"
        ][
            "quality_score"
        ]
    )

    print(
        "Risk Level:",
        result[
            "prediction"
        ][
            "risk_level"
        ]
    )


    print("\nTOP MODEL DRIVERS")
    print("=" * 70)

    for driver in (
        result[
            "explanation"
        ][
            "top_model_drivers"
        ]
    ):

        print(
            driver["feature"],
            "| observed:",
            driver["observed_value"],
            "| SHAP:",
            driver["shap_value"],
            "|",
            driver["direction"]
        )