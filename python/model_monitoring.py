import json
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_FILE = PROJECT_ROOT / "audit" / "analysis_audit.jsonl"
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "pharma_quality_data.csv"
)

OUTCOME_FILE = (
    PROJECT_ROOT
    / "audit"
    / "batch_outcomes.jsonl"
)

MIN_PERFORMANCE_SAMPLES = 30


def load_outcome_records():

    if not OUTCOME_FILE.exists():
        return pd.DataFrame()

    records = []

    with open(
        OUTCOME_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            records.append(record)

    return pd.DataFrame(records)

def calculate_model_performance():

    outcome_df = load_outcome_records()

    n_outcomes = len(outcome_df)

    if n_outcomes < MIN_PERFORMANCE_SAMPLES:

        return {
            "status": "INSUFFICIENT_DATA",
            "n_outcomes": n_outcomes,
            "minimum_required":
                MIN_PERFORMANCE_SAMPLES
        }

    actual = pd.to_numeric(
        outcome_df["actual_quality"],
        errors="coerce"
    )

    predicted = pd.to_numeric(
        outcome_df["predicted_quality"],
        errors="coerce"
    )

    valid = actual.notna() & predicted.notna()

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) < MIN_PERFORMANCE_SAMPLES:

        return {
            "status": "INSUFFICIENT_DATA",
            "n_outcomes": len(actual),
            "minimum_required":
                MIN_PERFORMANCE_SAMPLES
        }

    errors = actual - predicted

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else None
    )

    return {
        "status": "AVAILABLE",
        "n_outcomes": len(actual),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": (
            round(float(r2), 4)
            if r2 is not None
            else None
        )
    }
# ---------------------------------------------------------
# Load monitoring records
# ---------------------------------------------------------

def load_audit_records():

    if not AUDIT_FILE.exists():
        return pd.DataFrame()

    records = []

    with open(AUDIT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Monitoring requires production input features.
            # Older audit records may not contain them.
            if not record.get("input_features"):
                continue

            records.append(record)

    return pd.DataFrame(records)

# ---------------------------------------------------------
# PSI drift calculation
# ---------------------------------------------------------

MIN_MONITORING_SAMPLES = 30

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


def calculate_psi(reference, current, bins=10):

    reference = pd.to_numeric(
        reference,
        errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce"
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return None

    # Create bins from the historical reference distribution
    breakpoints = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(breakpoints) < 3:
        return None

    # Ensure new values outside historical min/max
    # are still captured.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts = np.histogram(
        reference,
        bins=breakpoints
    )[0]

    current_counts = np.histogram(
        current,
        bins=breakpoints
    )[0]

    reference_pct = (
        reference_counts / reference_counts.sum()
    )

    current_pct = (
        current_counts / current_counts.sum()
    )

    # Avoid log(0)
    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)
def calculate_feature_drift(audit_df):

    if len(audit_df) < MIN_MONITORING_SAMPLES:

        return {
            "status": "INSUFFICIENT_DATA",
            "n_records": len(audit_df),
            "minimum_required": MIN_MONITORING_SAMPLES,
            "features": {}
        }

    reference_df = pd.read_csv(DATA_FILE)

    # Convert nested input_features dictionaries
    # into normal columns.
    production_df = pd.json_normalize(
        audit_df["input_features"]
    )

    feature_results = {}

    for feature in NUMERIC_FEATURES:

        if feature not in production_df.columns:
            continue

        psi = calculate_psi(
            reference_df[feature],
            production_df[feature]
        )

        if psi is None:
            continue

        if psi > 0.25:
            status = "DRIFT"

        elif psi >= 0.10:
            status = "WARNING"

        else:
            status = "STABLE"

        feature_results[feature] = {
            "psi": round(psi, 4),
            "status": status
        }

    overall_status = "STABLE"

    statuses = [
        result["status"]
        for result in feature_results.values()
    ]

    if "DRIFT" in statuses:
        overall_status = "DRIFT"

    elif "WARNING" in statuses:
        overall_status = "WARNING"

    return {
        "status": overall_status,
        "n_records": len(audit_df),
        "minimum_required": MIN_MONITORING_SAMPLES,
        "features": feature_results
    }
# ---------------------------------------------------------
# Monitoring summary
# ---------------------------------------------------------

def summarize_monitoring():

    audit_df = load_audit_records()

    if audit_df.empty:

        return {
            "status": "NO_DATA",
            "n_records": 0
        }

    n_records = len(audit_df)

    # Applicability distribution
    applicability_counts = (
        audit_df["applicability_status"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_dict()
    )

    ood_count = applicability_counts.get(
        "OUT_OF_DISTRIBUTION",
        0
    )

    borderline_count = applicability_counts.get(
        "BORDERLINE",
        0
    )

    ood_rate = ood_count / n_records
    borderline_rate = borderline_count / n_records

    # Risk distribution
    risk_counts = (
        audit_df["risk_level"]
        .fillna("UNKNOWN")
        .value_counts()
        .to_dict()
    )

    # Prediction summary
    predicted_quality = pd.to_numeric(
        audit_df["predicted_quality"],
        errors="coerce"
    )

    prediction_summary = {
        "mean": round(predicted_quality.mean(), 4),
        "std": round(predicted_quality.std(), 4),
        "min": round(predicted_quality.min(), 4),
        "max": round(predicted_quality.max(), 4)
    }

    # Simple monitoring status
    if ood_rate >= 0.20:
        monitoring_status = "ALERT"

    elif (
        ood_rate >= 0.05
        or borderline_rate >= 0.20
    ):
        monitoring_status = "WARNING"

    else:
        monitoring_status = "STABLE"

    feature_drift = calculate_feature_drift(
        audit_df
    )

    model_performance = (
    calculate_model_performance()
    )
    return {
        "status": monitoring_status,
        "n_records": n_records,
        "applicability_counts": applicability_counts,
        "ood_rate": round(ood_rate, 4),
        "borderline_rate": round(borderline_rate, 4),
        "risk_counts": risk_counts,
        "prediction_summary": prediction_summary,
        "feature_drift": feature_drift,
        "model_performance": model_performance
    }


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":

    summary = summarize_monitoring()

    print("\nMODEL MONITORING SUMMARY")
    print("=" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")