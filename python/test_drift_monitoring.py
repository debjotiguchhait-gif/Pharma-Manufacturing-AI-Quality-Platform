import pandas as pd

from model_monitoring import (
    DATA_FILE,
    NUMERIC_FEATURES,
    calculate_psi
)


# ---------------------------------------------------------
# Load historical reference data
# ---------------------------------------------------------

reference_df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# TEST 1 — Production similar to training data
# ---------------------------------------------------------

normal_production = reference_df.sample(
    n=100,
    random_state=42
).copy()


# ---------------------------------------------------------
# TEST 2 — Deliberately shifted production
# ---------------------------------------------------------

shifted_production = reference_df.sample(
    n=100,
    random_state=42
).copy()

# Simulate manufacturing/process shift
shifted_production["Temperature_C"] += 4.0

shifted_production[
    "Intermediate_Impurity_pct"
] += 1.5

shifted_production[
    "Raw_Material_Purity_pct"
] -= 2.0


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def evaluate_population(name, production_df):

    print(f"\n{name}")
    print("=" * 65)

    for feature in NUMERIC_FEATURES:

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

        print(
            f"{feature:<32}"
            f"PSI = {psi:.4f}   "
            f"{status}"
        )


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

evaluate_population(
    "TEST 1 — NORMAL PRODUCTION",
    normal_production
)

evaluate_population(
    "TEST 2 — SHIFTED PRODUCTION",
    shifted_production
)