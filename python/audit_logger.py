import json
import uuid
from datetime import datetime
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

AUDIT_DIR = (
    PROJECT_ROOT / "audit"
)

AUDIT_FILE = (
    AUDIT_DIR / "analysis_audit.jsonl"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def generate_run_id():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    short_uuid = str(
        uuid.uuid4()
    )[:8]

    return (
        f"RUN_{timestamp}_{short_uuid}"
    )


# ============================================================
# AUDIT LOGGER
# ============================================================

def log_analysis(
    result,
    analysis_mode,
    gemini_enabled,
    input_features=None
):

    """
    Write one analysis event to JSONL.

    Logging failure should not break the
    prediction pipeline.
    """

    try:

        run_id = generate_run_id()

        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = (
            result.get("prediction")
            or {}
        )


        # ----------------------------------------------------
        # Applicability
        # ----------------------------------------------------

        validation = (
            result.get("validation")
            or {}
        )

        applicability = (
            validation.get(
                "overall_status"
            )
            if validation
            else "NOT_APPLICABLE"
        )


        # ----------------------------------------------------
        # SHAP drivers
        # ----------------------------------------------------

        drivers = []

        for driver in result.get(
            "top_model_drivers",
            []
        ):

            drivers.append({

                "feature":
                    driver.get(
                        "feature"
                    ),

                "observed_value":
                    driver.get(
                        "observed_value"
                    ),

                "shap_value":
                    driver.get(
                        "shap_value"
                    ),

                "direction":
                    driver.get(
                        "direction"
                    )
            })


        # ----------------------------------------------------
        # Retrieved knowledge
        # ----------------------------------------------------

        retrieved_kb = []

        for item in result.get(
            "retrieved_knowledge",
            []
        ):

            retrieved_kb.append({

                "kb_id":
                    item.get(
                        "kb_id"
                    ),

                "title":
                    item.get(
                        "title"
                    ),

                "retrieval_type":
                    item.get(
                        "retrieval_type",
                        "contextual"
                    )
            })


        # ----------------------------------------------------
        # Audit record
        # ----------------------------------------------------

        audit_record = {

            "run_id":
                run_id,

            "timestamp":
                timestamp,

            "analysis_mode":
                analysis_mode,

            "batch_id":
                result.get(
                    "batch_id"
                ),

            "input_features":
                input_features,

            "question":
                result.get(
                    "question"
                ),

            "applicability_status":
                applicability,

            "predicted_quality":
                prediction.get(
                    "quality_score"
                ),

            "risk_level":
                prediction.get(
                    "risk_level"
                ),

            "top_model_drivers":
                drivers,

            "retrieved_knowledge":
                retrieved_kb,

            "gemini_enabled":
                bool(
                    gemini_enabled
                ),

            "gemini_response_generated":
                bool(
                    result.get(
                        "ai_answer"
                    )
                )
        }


        # ----------------------------------------------------
        # Write JSONL
        # ----------------------------------------------------

        with open(
            AUDIT_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    audit_record,
                    ensure_ascii=False,
                    default=str
                )
                + "\n"
            )


        return {
            "success": True,
            "run_id": run_id,
            "timestamp": timestamp,
            "audit_file": str(
                AUDIT_FILE
            )
        }


    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }