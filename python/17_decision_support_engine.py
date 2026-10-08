from prediction_explanation_tool import (
    explain_batch,
    df,
    feature_columns
)

from hybrid_rag_retrieval import (
    retrieve_knowledge
)

from gemini_evidence_synthesis import (
    synthesize_answer
)


# ============================================================
# BATCH VALIDATION
# ============================================================

def get_batch_data(batch_id):

    """
    Find a batch in the historical dataset and
    return its model input features.
    """

    matching_rows = df[
        df["Batch_ID"] == batch_id
    ]

    if matching_rows.empty:

        raise ValueError(
            f"Batch ID not found: {batch_id}"
        )

    row = matching_rows.iloc[0]

    batch_data = (
        row[
            feature_columns
        ]
        .to_dict()
    )

    return batch_data


# ============================================================
# DECISION-SUPPORT ENGINE
# ============================================================

def analyze_batch(
    batch_id,
    question,
    top_drivers=5,
    contextual_k=3,
    use_gemini=True
):

    """
    Complete pharmaceutical manufacturing
    decision-support pipeline.

    Pipeline:
        Batch
          ↓
        ML Prediction
          ↓
        SHAP Explanation
          ↓
        Hybrid RAG
          ↓
        Mandatory Guardrails
          ↓
        Gemini Synthesis
    """

    # --------------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------------

    if not batch_id:

        raise ValueError(
            "Batch ID is required."
        )

    if not question:

        raise ValueError(
            "A user question is required."
        )


    # --------------------------------------------------------
    # 2. Load batch
    # --------------------------------------------------------

    batch_data = get_batch_data(
        batch_id
    )


    # --------------------------------------------------------
    # 3. ML prediction + SHAP
    # --------------------------------------------------------

    prediction_evidence = explain_batch(
        batch_data=batch_data,
        batch_id=batch_id,
        top_n=top_drivers
    )


    # --------------------------------------------------------
    # 4. Hybrid RAG + mandatory policy retrieval
    # --------------------------------------------------------

    retrieved_knowledge = retrieve_knowledge(
        query=question,
        top_k=contextual_k
    )


    # --------------------------------------------------------
    # 5. Gemini synthesis
    # --------------------------------------------------------

    if use_gemini:

        ai_answer = synthesize_answer(
            user_question=question,
            prediction_evidence=
                prediction_evidence,
            retrieved_knowledge=
                retrieved_knowledge
        )

    else:

        ai_answer = None


    # --------------------------------------------------------
    # 6. Structured application response
    # --------------------------------------------------------

    result = {

        "batch_id":
            batch_id,

        "question":
            question,

        "prediction":
            prediction_evidence[
                "prediction"
            ],

        "top_model_drivers":
            prediction_evidence[
                "explanation"
            ][
                "top_model_drivers"
            ],

        "guardrails":
            prediction_evidence.get(
                "guardrails",
                []
            ),

        "retrieved_knowledge":
            retrieved_knowledge,

        "ai_answer":
            ai_answer
    }


    return result


# ============================================================
# CONSOLE DISPLAY
# ============================================================

def display_result(result):

    """
    Simple console representation of the
    integrated engine output.
    """

    print("\n")
    print("=" * 80)
    print("PHARMA MANUFACTURING AI DECISION-SUPPORT ENGINE")
    print("=" * 80)


    print(
        "\nBatch:",
        result["batch_id"]
    )


    print(
        "Question:",
        result["question"]
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = result[
        "prediction"
    ]


    print("\n")
    print("PREDICTION")
    print("-" * 80)


    print(
        "Predicted Quality:",
        prediction[
            "quality_score"
        ]
    )


    print(
        "Risk Level:",
        prediction[
            "risk_level"
        ]
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\n")
    print("TOP MODEL DRIVERS")
    print("-" * 80)


    for driver in result[
        "top_model_drivers"
    ]:

        print(
            driver["feature"],
            "| observed:",
            driver["observed_value"],
            "| SHAP:",
            driver["shap_value"],
            "|",
            driver["direction"]
        )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    print("\n")
    print("RETRIEVED KNOWLEDGE")
    print("-" * 80)


    for rank, item in enumerate(
        result["retrieved_knowledge"],
        start=1
    ):

        print(
            f"{rank}.",
            item["kb_id"],
            "-",
            item["title"],
            "|",
            item.get(
                "retrieval_type",
                "contextual"
            )
        )


    # --------------------------------------------------------
    # AI response
    # --------------------------------------------------------

    if result["ai_answer"]:

        print("\n")
        print("AI DECISION-SUPPORT RESPONSE")
        print("-" * 80)

        print(
            result["ai_answer"]
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    TEST_BATCH = (
        "BATCH_11325"
    )


    TEST_QUESTION = (
        "Why is this batch predicted to have "
        "poor quality and what should I "
        "investigate?"
    )


    result = analyze_batch(

        batch_id=
            TEST_BATCH,

        question=
            TEST_QUESTION,

        top_drivers=5,

        contextual_k=3,

        use_gemini=False
    )


    display_result(
        result
    )