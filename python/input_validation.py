import pandas as pd
from applicability_model import (
    check_multivariate_applicability
)
from prediction_explanation_tool import (
    df,
    feature_columns
)


# ============================================================
# CONFIGURATION
# ============================================================

# Inner reference range:
# 1st–99th percentile
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99


# ============================================================
# FEATURE GROUPS
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


CATEGORICAL_FEATURES = [
    "Product",
    "Reactor",
    "Shift",
    "Supplier"
]


# ============================================================
# BUILD REFERENCE PROFILE
# ============================================================

def build_reference_profile():

    profile = {
        "numeric": {},
        "categorical": {}
    }


    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        series = (
            df[feature]
            .dropna()
            .astype(float)
        )

        profile["numeric"][feature] = {

            "min":
                float(series.min()),

            "q01":
                float(
                    series.quantile(
                        LOWER_QUANTILE
                    )
                ),

            "median":
                float(series.median()),

            "q99":
                float(
                    series.quantile(
                        UPPER_QUANTILE
                    )
                ),

            "max":
                float(series.max())
        }


    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        categories = (
            df[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        profile["categorical"][
            feature
        ] = sorted(categories)


    return profile


REFERENCE_PROFILE = (
    build_reference_profile()
)


# ============================================================
# VALIDATE NEW BATCH
# ============================================================

def validate_new_batch(
    batch_data
):

    """
    Validate an unseen manufacturing batch.

    Numeric logic:

    q01 <= value <= q99
        → IN_DISTRIBUTION

    historical min <= value < q01
    OR
    q99 < value <= historical max
        → BORDERLINE

    value < historical min
    OR
    value > historical max
        → OUT_OF_DISTRIBUTION

    Unknown categorical values are considered
    OUT_OF_DISTRIBUTION.
    """


    issues = []

    feature_results = {}

    # ========================================================
    # 1. SCHEMA CHECK
    # ========================================================

    missing_features = [

        feature

        for feature in feature_columns

        if feature not in batch_data
    ]


    if missing_features:

        return {

            "overall_status":
                "INVALID",

            "prediction_allowed":
                False,

            "issues": [
                (
                    "Missing required features: "
                    + ", ".join(
                        missing_features
                    )
                )
            ],

            "feature_results":
                {}
        }


    # ========================================================
    # 2. NUMERIC FEATURES
    # ========================================================

    for feature in NUMERIC_FEATURES:

        value = batch_data[
            feature
        ]


        reference = (
            REFERENCE_PROFILE[
                "numeric"
            ][
                feature
            ]
        )


        # ----------------------------------------------------
        # Missing value
        # ----------------------------------------------------

        if pd.isna(value):

            feature_results[
                feature
            ] = {

                "value":
                    None,

                "status":
                    "MISSING",

                "reference":
                    reference
            }


            issues.append(
                f"{feature}: missing value"
            )

            continue


        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            feature_results[
                feature
            ] = {

                "value":
                    value,

                "status":
                    "INVALID_TYPE",

                "reference":
                    reference
            }


            issues.append(
                f"{feature}: invalid numeric value"
            )

            continue


        # ----------------------------------------------------
        # OOD classification
        # ----------------------------------------------------

        if (
            value < reference["min"]
            or
            value > reference["max"]
        ):

            status = (
                "OUT_OF_DISTRIBUTION"
            )


            issues.append(
                f"{feature}: {value} is outside "
                f"historical range "
                f"[{reference['min']:.4f}, "
                f"{reference['max']:.4f}]"
            )


        elif (
            value < reference["q01"]
            or
            value > reference["q99"]
        ):

            status = (
                "BORDERLINE"
            )


            issues.append(
                f"{feature}: {value} is in a "
                "historically sparse region"
            )


        else:

            status = (
                "IN_DISTRIBUTION"
            )


        feature_results[
            feature
        ] = {

            "value":
                value,

            "status":
                status,

            "reference":
                reference
        }


    # ========================================================
    # 3. CATEGORICAL FEATURES
    # ========================================================

    for feature in CATEGORICAL_FEATURES:

        value = batch_data[
            feature
        ]


        allowed_values = (
            REFERENCE_PROFILE[
                "categorical"
            ][
                feature
            ]
        )


        if pd.isna(value):

            status = "MISSING"

            issues.append(
                f"{feature}: missing value"
            )


        elif str(value) not in (
            allowed_values
        ):

            status = (
                "OUT_OF_DISTRIBUTION"
            )


            issues.append(
                f"{feature}: unseen category "
                f"'{value}'"
            )


        else:

            status = (
                "IN_DISTRIBUTION"
            )


        feature_results[
            feature
        ] = {

            "value":
                value,

            "status":
                status,

            "allowed_values":
                allowed_values
        }


    # ========================================================
    # 4. DETERMINE OVERALL STATUS
    # ========================================================

    statuses = [

        result["status"]

        for result
        in feature_results.values()
    ]


    if (
        "INVALID_TYPE"
        in statuses
        or
        "MISSING"
        in statuses
    ):

        overall_status = (
            "INVALID"
        )

        prediction_allowed = False


    elif (
        "OUT_OF_DISTRIBUTION"
        in statuses
    ):

        overall_status = (
            "OUT_OF_DISTRIBUTION"
        )

        # We allow prediction for demonstration,
        # but it must carry a strong reliability
        # warning.

        prediction_allowed = True


    elif (
        "BORDERLINE"
        in statuses
    ):

        overall_status = (
            "BORDERLINE"
        )

        prediction_allowed = True


    else:

        overall_status = (
            "IN_DISTRIBUTION"
        )

        prediction_allowed = True


    # ========================================================
    # 5. RETURN VALIDATION REPORT
    # ========================================================

    return {

        "overall_status":
            overall_status,

        "prediction_allowed":
            prediction_allowed,

        "issues":
            issues,

        "feature_results":
            feature_results
    }

# ============================================================
# COMPLETE APPLICABILITY ASSESSMENT
# ============================================================

def assess_model_applicability(
    batch_data
):

    """
    Combine:

    1. Schema / datatype validation
    2. Univariate applicability
    3. Multivariate applicability

    into one model-applicability assessment.
    """

    # --------------------------------------------------------
    # 1. Univariate validation
    # --------------------------------------------------------

    univariate = validate_new_batch(
        batch_data
    )


    # Invalid data should never reach the model
    if not univariate[
        "prediction_allowed"
    ]:

        return {

            "overall_status":
                "INVALID",

            "prediction_allowed":
                False,

            "univariate":
                univariate,

            "multivariate":
                None,

            "issues":
                univariate[
                    "issues"
                ]
        }


    # --------------------------------------------------------
    # 2. Multivariate applicability
    # --------------------------------------------------------

    multivariate = (
        check_multivariate_applicability(
            batch_data
        )
    )


    uni_status = univariate[
        "overall_status"
    ]

    multi_status = multivariate[
        "status"
    ]


    # --------------------------------------------------------
    # 3. Final applicability decision
    # --------------------------------------------------------

    if (
        uni_status
        == "OUT_OF_DISTRIBUTION"
        or
        multi_status
        == "OUT_OF_DISTRIBUTION"
    ):

        overall_status = (
            "OUT_OF_DISTRIBUTION"
        )


    elif (
        uni_status
        == "BORDERLINE"
        or
        multi_status
        == "BORDERLINE"
    ):

        overall_status = (
            "BORDERLINE"
        )


    else:

        overall_status = (
            "IN_DISTRIBUTION"
        )


    # --------------------------------------------------------
    # 4. Consolidated issues
    # --------------------------------------------------------

    issues = list(
        univariate[
            "issues"
        ]
    )


    if multi_status == (
        "OUT_OF_DISTRIBUTION"
    ):

        issues.append(
            "The combined numeric feature pattern "
            "is outside the historical multivariate "
            "applicability domain."
        )


    elif multi_status == (
        "BORDERLINE"
    ):

        issues.append(
            "The combined numeric feature pattern "
            "is in a historically sparse "
            "multivariate region."
        )


    return {

        "overall_status":
            overall_status,

        "prediction_allowed":
            True,

        "univariate":
            univariate,

        "multivariate":
            multivariate,

        "issues":
            issues
    }

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Use medians / known categories to create
    # a normal test batch.

    test_batch = {

        "Product":
            "Product_A",

        "Reactor":
            "Reactor_01",

        "Temperature_C":
            50.0,

        "pH":
            6.5,

        "Agitation_rpm":
            300.0,

        "Process_Time_hr":
            8.0,

        "Pressure_bar":
            2.0,

        "Raw_Material_Purity_pct":
            99.0,

        "Operator_Experience_yr":
            5.0,

        "Equipment_Age_yr":
            5.0,

        "Room_Humidity_pct":
            45.0,

        "Intermediate_Impurity_pct":
            2.0,

        "Shift":
            "Morning",

        "Supplier":
            "Supplier_X"
    }


    result = validate_new_batch(
        test_batch
    )


    print(
        "\nNEW BATCH VALIDATION"
    )

    print("=" * 70)


    print(
        "Overall Status:",
        result[
            "overall_status"
        ]
    )


    print(
        "Prediction Allowed:",
        result[
            "prediction_allowed"
        ]
    )


    print(
        "\nFEATURE VALIDATION"
    )

    print("-" * 70)


    for feature, details in (
        result[
            "feature_results"
        ].items()
    ):

        print(
            feature,
            "|",
            details["value"],
            "|",
            details["status"]
        )


    if result["issues"]:

        print(
            "\nISSUES / WARNINGS"
        )

        print("-" * 70)

        for issue in result[
            "issues"
        ]:

            print(
                "-",
                issue
            )