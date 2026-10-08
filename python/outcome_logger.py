import json
from datetime import datetime
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIT_FILE = (
    PROJECT_ROOT
    / "audit"
    / "analysis_audit.jsonl"
)

OUTCOME_FILE = (
    PROJECT_ROOT
    / "audit"
    / "batch_outcomes.jsonl"
)


# ============================================================
# FIND AUDIT RECORD
# ============================================================

def find_audit_record(run_id):

    if not AUDIT_FILE.exists():
        return None

    with open(
        AUDIT_FILE,
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

            if record.get("run_id") == run_id:
                return record

    return None


# ============================================================
# CHECK EXISTING OUTCOME
# ============================================================

def outcome_exists(run_id):

    if not OUTCOME_FILE.exists():
        return False

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

            if record.get("run_id") == run_id:
                return True

    return False


# ============================================================
# RECORD ACTUAL OUTCOME
# ============================================================

def record_actual_outcome(
    run_id,
    actual_quality
):

    audit_record = find_audit_record(
        run_id
    )

    if audit_record is None:

        return {
            "success": False,
            "error": "RUN_ID_NOT_FOUND"
        }

    if outcome_exists(run_id):

        return {
            "success": False,
            "error": "OUTCOME_ALREADY_RECORDED"
        }

    try:
        actual_quality = float(
            actual_quality
        )

    except (TypeError, ValueError):

        return {
            "success": False,
            "error": "INVALID_ACTUAL_QUALITY"
        }

    predicted_quality = audit_record.get(
        "predicted_quality"
    )

    if predicted_quality is None:

        return {
            "success": False,
            "error": "PREDICTION_NOT_AVAILABLE"
        }

    predicted_quality = float(
        predicted_quality
    )

    prediction_error = (
        actual_quality
        - predicted_quality
    )

    absolute_error = abs(
        prediction_error
    )

    outcome_record = {

        "run_id":
            run_id,

        "batch_id":
            audit_record.get(
                "batch_id"
            ),

        "prediction_timestamp":
            audit_record.get(
                "timestamp"
            ),

        "outcome_timestamp":
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            ),

        "predicted_quality":
            round(
                predicted_quality,
                4
            ),

        "actual_quality":
            round(
                actual_quality,
                4
            ),

        "prediction_error":
            round(
                prediction_error,
                4
            ),

        "absolute_error":
            round(
                absolute_error,
                4
            )
    }

    with open(
        OUTCOME_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                outcome_record,
                ensure_ascii=False
            )
            + "\n"
        )

    return {
        "success": True,
        "outcome": outcome_record
    }

if __name__ == "__main__":

    result = record_actual_outcome(
        run_id="RUN_20261005_232547_a8f3615f",
        actual_quality=89.80
    )

    print(result)