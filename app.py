import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = PROJECT_ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIR)
    )


# ============================================================
# IMPORT BACKEND
# ============================================================

from decision_support_engine import (
    analyze_batch,
    analyze_new_batch,
    df
)

from input_validation import (
    REFERENCE_PROFILE
)

from model_monitoring import (
    summarize_monitoring
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pharma Manufacturing AI Quality Platform",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Pharma Manufacturing AI Quality Platform")

st.markdown(
    """
    **AI-enabled manufacturing quality decision support**

    Predict batch quality, assess model applicability, explain predictions,
    retrieve process knowledge, and monitor deployed-model behavior.
    """
)

st.info(
    "Decision-support prototype — outputs support investigation and "
    "process understanding and are not validated pharmaceutical "
    "release or batch-disposition criteria."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Analysis Settings"
    )

    use_gemini = st.toggle(
        "Enable Gemini synthesis",
        value=False
    )

    top_drivers = st.slider(
        "Number of SHAP drivers",
        min_value=3,
        max_value=10,
        value=5
    )

    contextual_k = st.slider(
        "Contextual RAG documents",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    st.caption(
        "Gemini is used only for grounded evidence "
        "synthesis. Prediction, SHAP, validation and "
        "retrieval are handled separately."
    )


# ============================================================
# ANALYSIS MODE
# ============================================================

st.subheader("Workspace")

analysis_mode = st.radio(
    "Analysis Mode",
    options=[
        "Historical Batch",
        "New Batch",
        "Model Monitoring"
    ],
    horizontal=True
)


# ============================================================
# HISTORICAL BATCH MODE
# ============================================================

if analysis_mode == "Historical Batch":

    batch_ids = sorted(
        df["Batch_ID"]
        .astype(str)
        .unique()
        .tolist()
    )

    default_batch = (
        "BATCH_11325"
        if "BATCH_11325" in batch_ids
        else batch_ids[0]
    )

    default_index = batch_ids.index(
        default_batch
    )

    batch_id = st.selectbox(
        "Select Batch ID",
        options=batch_ids,
        index=default_index
    )

    question = st.text_area(
        "Ask a manufacturing question",
        value=(
            "Why is this batch predicted to have "
            "poor quality and what should I investigate?"
        ),
        height=100
    )

    analyze_button = st.button(
        "Analyze Historical Batch",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not question.strip():

            st.error(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Running prediction, SHAP, retrieval "
                    "and decision support..."
                ):

                    result = analyze_batch(
                        batch_id=batch_id,
                        question=question.strip(),
                        top_drivers=top_drivers,
                        contextual_k=contextual_k,
                        use_gemini=use_gemini
                    )

                st.session_state[
                    "analysis_result"
                ] = result

                st.session_state[
                    "analysis_mode_result"
                ] = "Historical Batch"

            except Exception as error:

                st.error(
                    "Analysis failed."
                )

                st.exception(
                    error
                )


# ============================================================
# NEW BATCH MODE
# ============================================================

elif analysis_mode == "New Batch":

    st.info(
        "Enter process information for an unseen batch. "
        "The system will validate model applicability "
        "before generating a prediction."
    )

    batch_id = st.text_input(
        "New Batch ID",
        value="NEW_BATCH_001"
    )


    # --------------------------------------------------------
    # Categorical inputs
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        product = st.selectbox(
            "Product",
            REFERENCE_PROFILE[
                "categorical"
            ][
                "Product"
            ]
        )

        reactor = st.selectbox(
            "Reactor",
            REFERENCE_PROFILE[
                "categorical"
            ][
                "Reactor"
            ]
        )

    with col2:

        shift = st.selectbox(
            "Shift",
            REFERENCE_PROFILE[
                "categorical"
            ][
                "Shift"
            ]
        )

        supplier = st.selectbox(
            "Supplier",
            REFERENCE_PROFILE[
                "categorical"
            ][
                "Supplier"
            ]
        )


    # --------------------------------------------------------
    # Numeric inputs — row 1
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        temperature = st.number_input(
            "Temperature (°C)",
            value=26.0,
            format="%.2f"
        )

    with col2:

        ph = st.number_input(
            "pH",
            value=6.5,
            format="%.2f"
        )

    with col3:

        agitation = st.number_input(
            "Agitation (rpm)",
            value=300.0,
            format="%.2f"
        )


    # --------------------------------------------------------
    # Numeric inputs — row 2
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        process_time = st.number_input(
            "Process Time (hr)",
            value=8.0,
            format="%.2f"
        )

    with col2:

        pressure = st.number_input(
            "Pressure (bar)",
            value=2.0,
            format="%.2f"
        )

    with col3:

        raw_purity = st.number_input(
            "Raw Material Purity (%)",
            value=99.0,
            format="%.2f"
        )


    # --------------------------------------------------------
    # Numeric inputs — row 3
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        operator_experience = st.number_input(
            "Operator Experience (yr)",
            value=5.0,
            format="%.2f"
        )

    with col2:

        equipment_age = st.number_input(
            "Equipment Age (yr)",
            value=5.0,
            format="%.2f"
        )

    with col3:

        humidity = st.number_input(
            "Room Humidity (%)",
            value=45.0,
            format="%.2f"
        )


    # --------------------------------------------------------
    # Numeric inputs — row 4
    # --------------------------------------------------------

    impurity = st.number_input(
        "Intermediate Impurity (%)",
        value=2.0,
        format="%.3f"
    )


    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = st.text_area(
        "Ask a manufacturing question",
        value=(
            "What does the model predict for this "
            "new batch and what should I investigate?"
        ),
        height=100
    )


    # --------------------------------------------------------
    # Build new batch
    # --------------------------------------------------------

    new_batch = {

        "Product":
            product,

        "Reactor":
            reactor,

        "Temperature_C":
            temperature,

        "pH":
            ph,

        "Agitation_rpm":
            agitation,

        "Process_Time_hr":
            process_time,

        "Pressure_bar":
            pressure,

        "Raw_Material_Purity_pct":
            raw_purity,

        "Operator_Experience_yr":
            operator_experience,

        "Equipment_Age_yr":
            equipment_age,

        "Room_Humidity_pct":
            humidity,

        "Intermediate_Impurity_pct":
            impurity,

        "Shift":
            shift,

        "Supplier":
            supplier
    }


    analyze_button = st.button(
        "Validate & Analyze New Batch",
        type="primary",
        use_container_width=True
    )


    if analyze_button:

        try:

            with st.spinner(
                "Validating applicability and "
                "running prediction..."
            ):

                result = analyze_new_batch(
                    batch_data=new_batch,
                    batch_id=batch_id,
                    question=(
                        question.strip()
                        if question.strip()
                        else None
                    ),
                    top_drivers=top_drivers,
                    contextual_k=contextual_k,
                    use_gemini=use_gemini
                )

            st.session_state[
                "analysis_result"
            ] = result

            st.session_state[
                "analysis_mode_result"
            ] = "New Batch"

        except Exception as error:

            st.error(
                "New batch analysis failed."
            )

            st.exception(
                error
            )
# ============================================================
# MODEL MONITORING MODE
# ============================================================

elif analysis_mode == "Model Monitoring":

    st.header("Model Monitoring")

    monitoring = summarize_monitoring()

    n_records = monitoring.get(
        "n_records",
        0
    )

    st.metric(
        "Monitored Batches",
        n_records
    )

    # --------------------------------------------------------
    # Operational monitoring
    # --------------------------------------------------------

    st.subheader(
        "Operational Status"
    )

    status = monitoring.get(
        "status",
        "NO_DATA"
    )

    if status == "STABLE":
        st.success("STABLE")

    elif status == "WARNING":
        st.warning("WARNING")

    elif status == "ALERT":
        st.error("ALERT")

    else:
        st.info(status)

    if n_records > 0:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "OOD Rate",
                f"{monitoring['ood_rate'] * 100:.1f}%"
            )

        with col2:

            st.metric(
                "Borderline Rate",
                f"{monitoring['borderline_rate'] * 100:.1f}%"
            )

        # ----------------------------------------------------
        # Monitoring summary
        # ----------------------------------------------------

        st.markdown("#### Batch Distribution")

        applicability = monitoring.get(
            "applicability_counts",
            {}
        )

        risk = monitoring.get(
            "risk_counts",
            {}
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "In Distribution",
                applicability.get(
                    "IN_DISTRIBUTION",
                    0
                )
            )

        with col2:
            st.metric(
                "Borderline",
                applicability.get(
                    "BORDERLINE",
                    0
                )
            )

        with col3:
            st.metric(
                "Out of Distribution",
                applicability.get(
                    "OUT_OF_DISTRIBUTION",
                    0
                )
            )


        st.markdown("#### Risk Distribution")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Low Risk",
                risk.get("LOW", 0)
            )

        with col2:
            st.metric(
                "Medium Risk",
                risk.get("MEDIUM", 0)
            )

        with col3:
            st.metric(
                "High Risk",
                risk.get("HIGH", 0)
            )


        st.markdown("#### Prediction Summary")

        prediction_summary = monitoring.get(
            "prediction_summary",
            {}
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean Quality",
                f"{prediction_summary.get('mean', 0):.2f}"
            )

        with col2:
            st.metric(
                "Minimum Quality",
                f"{prediction_summary.get('min', 0):.2f}"
            )

        with col3:
            st.metric(
                "Maximum Quality",
                f"{prediction_summary.get('max', 0):.2f}"
            )

    # --------------------------------------------------------
    # Population drift
    # --------------------------------------------------------

    st.subheader(
        "Population Feature Drift"
    )

    drift = monitoring.get(
        "feature_drift",
        {}
    )

    drift_status = drift.get(
        "status",
        "NO_DATA"
    )

    if drift_status == "INSUFFICIENT_DATA":

        st.info(
            f"Insufficient production data for drift analysis. "
            f"Current records: {drift.get('n_records', 0)} | "
            f"Minimum required: "
            f"{drift.get('minimum_required', 30)}"
        )

    else:

        if drift_status == "STABLE":

            st.success(
                "Population drift status: STABLE"
            )

        elif drift_status == "WARNING":

            st.warning(
                "Population drift status: WARNING"
            )

        elif drift_status == "DRIFT":

            st.error(
                "Population drift status: DRIFT"
            )

        feature_results = drift.get(
            "features",
            {}
        )

        if feature_results:

            drift_df = pd.DataFrame(
                [
                    {
                        "Feature": feature,
                        "PSI": values["psi"],
                        "Status": values["status"]
                    }
                    for feature, values
                    in feature_results.items()
                ]
            )

            drift_df = drift_df.sort_values(
                by="PSI",
                ascending=False
            )

            st.dataframe(
                drift_df,
                use_container_width=True,
                hide_index=True
            )
    # --------------------------------------------------------
    # Model performance monitoring
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    performance = monitoring.get(
        "model_performance",
        {}
    )

    performance_status = performance.get(
        "status",
        "NO_DATA"
    )

    if performance_status == "INSUFFICIENT_DATA":

        current_outcomes = performance.get(
            "n_outcomes",
            0
        )

        minimum_required = performance.get(
            "minimum_required",
            30
        )

        st.info(
            f"Insufficient actual QC outcomes for "
            f"performance monitoring. "
            f"Current outcomes: {current_outcomes} | "
            f"Minimum required: {minimum_required}"
        )

        st.write(
            "**Development Holdout Baseline**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Baseline R²",
                "0.8772"
            )

        with col2:
            st.metric(
                "Baseline MAE",
                "1.4314"
            )

        with col3:
            st.metric(
                "Baseline RMSE",
                "1.8205"
            )

    elif performance_status == "AVAILABLE":

        st.success(
            "Production performance metrics available."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Production R²",
                performance["r2"]
            )

        with col2:
            st.metric(
                "Production MAE",
                performance["mae"]
            )

        with col3:
            st.metric(
                "Production RMSE",
                performance["rmse"]
            )

        st.caption(
            f"Calculated from "
            f"{performance['n_outcomes']} "
            f"batches with actual QC outcomes."
        )

        st.write(
            "**Development Holdout Baseline:** "
            "R² = 0.8772 | "
            "MAE = 1.4314 | "
            "RMSE = 1.8205"
        )
    st.caption(
        "Monitoring thresholds are demonstration-level "
        "engineering thresholds and are not validated GMP "
        "acceptance criteria."
    )

# ============================================================
# DISPLAY RESULT
# ============================================================

if (
    analysis_mode != "Model Monitoring"
    and "analysis_result" in st.session_state
):

    result = st.session_state[
        "analysis_result"
    ]

    result_mode = st.session_state.get(
        "analysis_mode_result"
    )


    # --------------------------------------------------------
    # Applicability — new batches only
    # --------------------------------------------------------

    if (
        result_mode == "New Batch"
        and
        "validation" in result
    ):

        st.divider()

        st.subheader(
            "Model Applicability"
        )

        validation = result[
            "validation"
        ]

        status = validation[
            "overall_status"
        ]


        if status == "IN_DISTRIBUTION":

            st.success(
                "IN DISTRIBUTION — Input values are "
                "within the model's historical "
                "reference domain."
            )


        elif status == "BORDERLINE":

            st.warning(
                "BORDERLINE — One or more inputs are "
                "in historically sparse regions. "
                "Interpret the prediction with "
                "additional caution."
            )


        elif status == "OUT_OF_DISTRIBUTION":

            st.error(
                "OUT OF DISTRIBUTION — One or more "
                "inputs fall outside the model's "
                "historical applicability domain. "
                "Prediction reliability may be reduced."
            )


        else:

            st.error(
                "INVALID INPUT — Prediction cannot "
                "be generated."
            )


        if validation[
            "issues"
        ]:

            with st.expander(
                "Validation warnings"
            ):

                for issue in validation[
                    "issues"
                ]:

                    st.write(
                        f"• {issue}"
                    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if result.get(
        "prediction"
    ) is not None:

        st.divider()

        st.subheader(
            "Batch Quality Assessment"
        )

        prediction = result[
            "prediction"
        ]

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Batch",
                result[
                    "batch_id"
                ]
            )

        with col2:

            st.metric(
                "Predicted Quality Score",
                f"{prediction['quality_score']:.2f}"
            )

        with col3:

            st.metric(
                "Risk Level",
                prediction[
                    "risk_level"
                ]
            )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.subheader(
            "Key Model Drivers"
        )

        st.caption(
        "Features with the largest contribution to this "
        "individual model prediction."
        )

        driver_rows = []

        for driver in result[
            "top_model_drivers"
        ]:

            driver_rows.append({

                "Feature":
                    driver[
                        "feature"
                    ],

                "Observed Value":
                    driver[
                        "observed_value"
                    ],

                "SHAP Contribution":
                    driver[
                        "shap_value"
                    ],

                "Direction":
                    driver[
                        "direction"
                    ]
            })


        driver_df = pd.DataFrame(
            driver_rows
        )

        st.dataframe(
            driver_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "SHAP values explain model behavior "
            "and do not establish scientific causation."
        )


        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        if result[
            "retrieved_knowledge"
        ]:

            st.subheader(
                "Process Knowledge & Guardrails"
            )

            for item in result[
                "retrieved_knowledge"
            ]:

                retrieval_type = (
                    item.get(
                        "retrieval_type",
                        "contextual"
                    )
                )

                if (
                    retrieval_type
                    ==
                    "mandatory_policy"
                ):

                    label = (
                        "Mandatory Policy"
                    )

                else:

                    label = (
                        "Contextual Evidence"
                    )


                with st.expander(
                    f"{item['kb_id']} — "
                    f"{item['title']} "
                    f"[{label}]"
                ):

                    st.write(
                        item[
                            "content"
                        ]
                    )


        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        st.subheader(
            "AI-Assisted Investigation Summary"
        )

        if result.get(
            "ai_answer"
        ):

            st.markdown(
                result[
                    "ai_answer"
                ]
            )

        else:

            st.info(
                "Gemini synthesis was disabled. "
                "Deterministic ML, SHAP, "
                "applicability and RAG evidence "
                "were generated successfully."
            )


    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Decision-Support Boundary"
    )

    st.info(
        "This platform supports investigation "
        "prioritization and process understanding. "
        "It does not independently establish root "
        "cause or make pharmaceutical batch release, "
        "rejection, approval or disposition decisions."
    )