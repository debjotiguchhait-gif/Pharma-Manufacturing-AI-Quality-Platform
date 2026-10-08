import numpy as np


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.array(
        actual,
        dtype=float
    )

    predicted = np.array(
        predicted,
        dtype=float
    )

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
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": (
            round(float(r2), 4)
            if r2 is not None
            else None
        )
    }


# ============================================================
# CREATE CONTROLLED TEST DATA
# ============================================================

rng = np.random.default_rng(42)

n_batches = 100

# Simulated actual production quality
actual_quality = rng.normal(
    loc=87.5,
    scale=5.0,
    size=n_batches
)


# ------------------------------------------------------------
# Scenario A — Healthy model
# ------------------------------------------------------------

healthy_prediction = (
    actual_quality
    + rng.normal(
        loc=0,
        scale=1.8,
        size=n_batches
    )
)


# ------------------------------------------------------------
# Scenario B — Degraded model
# ------------------------------------------------------------

degraded_prediction = (
    actual_quality
    + 3.0
    + rng.normal(
        loc=0,
        scale=4.5,
        size=n_batches
    )
)


# ============================================================
# EVALUATE
# ============================================================

healthy_metrics = calculate_metrics(
    actual_quality,
    healthy_prediction
)

degraded_metrics = calculate_metrics(
    actual_quality,
    degraded_prediction
)


print(
    "\nSCENARIO A — HEALTHY MODEL"
)
print("=" * 50)

for metric, value in healthy_metrics.items():
    print(
        f"{metric:<10}: {value}"
    )


print(
    "\nSCENARIO B — DEGRADED MODEL"
)
print("=" * 50)

for metric, value in degraded_metrics.items():
    print(
        f"{metric:<10}: {value}"
    )