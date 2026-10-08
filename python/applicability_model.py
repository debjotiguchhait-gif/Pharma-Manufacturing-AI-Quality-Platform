import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from prediction_explanation_tool import df


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Temperature_C",
    "pH",
    "Agitation_rpm",
    "Process_Time_hr",
    "Pressure_bar",
    "Raw_Material_Purity_pct",
    "Operator_Experience_yr",
    "Equipment_Age_yr",
    "Room_Humidity_pct",
    "Intermediate_Impurity_pct"
]


# ============================================================
# TRAIN REFERENCE APPLICABILITY MODEL
# ============================================================

X_reference = df[
    NUMERIC_FEATURES
].copy()


# Missing-value handling
imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(
    X_reference
)


# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)


# Isolation Forest
ood_model = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

ood_model.fit(
    X_scaled
)


# ============================================================
# REFERENCE SCORE DISTRIBUTION
# ============================================================

reference_scores = (
    ood_model.decision_function(
        X_scaled
    )
)


# Lower score = more unusual

SCORE_Q01 = float(
    np.quantile(
        reference_scores,
        0.01
    )
)

SCORE_Q05 = float(
    np.quantile(
        reference_scores,
        0.05
    )
)


# ============================================================
# MULTIVARIATE APPLICABILITY CHECK
# ============================================================

def check_multivariate_applicability(
    batch_data
):

    """
    Evaluate whether the combination of numeric
    process variables resembles the historical
    development distribution.

    This is an applicability indicator, not a
    quality or failure prediction.
    """

    input_row = pd.DataFrame(
        [
            {
                feature:
                    batch_data[
                        feature
                    ]

                for feature
                in NUMERIC_FEATURES
            }
        ]
    )


    # Same preprocessing as reference data
    input_imputed = imputer.transform(
        input_row
    )

    input_scaled = scaler.transform(
        input_imputed
    )


    # Isolation Forest score
    score = float(
        ood_model.decision_function(
            input_scaled
        )[0]
    )


    raw_prediction = int(
        ood_model.predict(
            input_scaled
        )[0]
    )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if raw_prediction == -1:

        status = (
            "OUT_OF_DISTRIBUTION"
        )

    elif score <= SCORE_Q05:

        status = (
            "BORDERLINE"
        )

    else:

        status = (
            "IN_DISTRIBUTION"
        )


    return {

        "status":
            status,

        "anomaly_score":
            round(
                score,
                6
            ),

        "reference_q01":
            round(
                SCORE_Q01,
                6
            ),

        "reference_q05":
            round(
                SCORE_Q05,
                6
            ),

        "interpretation":
            (
                "Lower anomaly scores indicate "
                "more unusual multivariate "
                "feature combinations."
            )
    }


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test 1: normal batch
    # --------------------------------------------------------

    normal_batch = {

        "Temperature_C": 26.0,
        "pH": 6.5,
        "Agitation_rpm": 300.0,
        "Process_Time_hr": 8.0,
        "Pressure_bar": 2.0,
        "Raw_Material_Purity_pct": 99.0,
        "Operator_Experience_yr": 5.0,
        "Equipment_Age_yr": 5.0,
        "Room_Humidity_pct": 45.0,
        "Intermediate_Impurity_pct": 2.0
    }


    # --------------------------------------------------------
    # Test 2: deliberately unusual combination
    # --------------------------------------------------------

    unusual_batch = {

        "Temperature_C": 34.5,
        "pH": 7.1,
        "Agitation_rpm": 480.0,
        "Process_Time_hr": 11.5,
        "Pressure_bar": 3.8,
        "Raw_Material_Purity_pct": 96.5,
        "Operator_Experience_yr": 1.0,
        "Equipment_Age_yr": 14.0,
        "Room_Humidity_pct": 68.0,
        "Intermediate_Impurity_pct": 4.5
    }


    print(
        "\nNORMAL BATCH"
    )

    print("=" * 70)

    print(
        check_multivariate_applicability(
            normal_batch
        )
    )


    print(
        "\nUNUSUAL BATCH"
    )

    print("=" * 70)

    print(
        check_multivariate_applicability(
            unusual_batch
        )
    )