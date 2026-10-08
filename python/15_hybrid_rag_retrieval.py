import re
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# PATHS
# ============================================

project_root = Path(__file__).resolve().parent.parent

knowledge_file = (
    project_root
    / "knowledge"
    / "process_knowledge.txt"
)


# ============================================
# LOAD KNOWLEDGE BASE
# ============================================

with open(
    knowledge_file,
    "r",
    encoding="utf-8"
) as file:

    knowledge_text = file.read()


# ============================================
# PARSE KB SECTIONS
# ============================================

pattern = (
    r"\[(KB-\d+)\]\s+([^\n]+)\n"
    r"(.*?)(?=\n\[KB-\d+\]|\Z)"
)

matches = re.findall(
    pattern,
    knowledge_text,
    flags=re.DOTALL
)


knowledge_chunks = []

for kb_id, title, content in matches:

    content = content.strip()

    knowledge_chunks.append({
        "kb_id": kb_id,
        "title": title.strip(),
        "content": content,
        "embedding_text":
            f"{title.strip()}\n{content}"
    })


print(
    f"\nLoaded {len(knowledge_chunks)} "
    "knowledge chunks."
)


# ============================================
# EMBEDDING MODEL
# ============================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


kb_texts = [
    chunk["embedding_text"]
    for chunk in knowledge_chunks
]


kb_embeddings = embedding_model.encode(
    kb_texts,
    normalize_embeddings=True
)


print("Knowledge embeddings ready.")


# ============================================
# INTENT RULES
# ============================================

intent_rules = {

    "KB-001": [
        "temperature",
        "thermal",
        "heating",
        "cooling"
    ],

    "KB-002": [
        "ph",
        "acidity",
        "alkalinity"
    ],

    "KB-003": [
        "impurity",
        "intermediate impurity"
    ],

    "KB-004": [
        "raw material",
        "raw-material",
        "material purity"
    ],

    "KB-005": [
        "reactor",
        "equipment"
    ],

    "KB-006": [
        "supplier",
        "vendor"
    ],

    "KB-007": [
        "process time",
        "hold time",
        "processing time"
    ],

    "KB-008": [
        "shap",
        "explain prediction",
        "explainability",
        "feature importance",
        "model contribution"
    ],

    "KB-010": [
        "cause",
        "caused",
        "causal",
        "causation",
        "root cause"
    ],

    "KB-011": [
        "investigate",
        "investigation",
        "what should i check"
    ],

    "KB-012": [
        "release batch",
        "reject batch",
        "approve batch",
        "batch disposition",
        "override specification"
    ]
}


# ============================================
# INTENT DETECTION
# ============================================
def contains_term(text, term):

    pattern = r"\b" + re.escape(term) + r"\b"

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )
def detect_intents(query):

    query_lower = query.lower()

    matched_kbs = {}

    for kb_id, keywords in intent_rules.items():

        matches = [
    keyword
    for keyword in keywords
    if contains_term(
        query_lower,
        keyword
    )
]
        if matches:

            matched_kbs[kb_id] = matches

    return matched_kbs


# ============================================
# HYBRID RETRIEVAL
# ============================================

def retrieve_knowledge(
    query,
    top_k=3,
    semantic_weight=0.75,
    intent_weight=0.25
):

    # ----------------------------------------
    # Semantic retrieval
    # ----------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )


    semantic_scores = cosine_similarity(
        query_embedding,
        kb_embeddings
    )[0]


    # ----------------------------------------
    # Intent detection
    # ----------------------------------------

    matched_intents = detect_intents(
        query
    )


    results = []


    for index, chunk in enumerate(
        knowledge_chunks
    ):

        kb_id = chunk["kb_id"]

        semantic_score = float(
            semantic_scores[index]
        )


        # Binary intent signal
        intent_score = (
            1.0
            if kb_id in matched_intents
            else 0.0
        )


        hybrid_score = (
            semantic_weight
            * semantic_score
            +
            intent_weight
            * intent_score
        )


        results.append({

            "kb_id":
                kb_id,

            "title":
                chunk["title"],

            "semantic_score":
                semantic_score,

            "intent_match":
                kb_id in matched_intents,

            "matched_keywords":
                matched_intents.get(
                    kb_id,
                    []
                ),

            "hybrid_score":
                hybrid_score,

            "content":
                chunk["content"]
        })


    # ----------------------------------------
    # Rank
    # ----------------------------------------

    results = sorted(
        results,
        key=lambda x:
            x["hybrid_score"],
        reverse=True
    )


        # ----------------------------------------
    # Mandatory guardrails
    # ----------------------------------------

    query_lower = query.lower()

    mandatory_kbs = []


    # Causality guardrail
    causality_terms = [
        "cause",
        "caused",
        "causal",
        "causation",
        "root cause"
    ]

    if any(
        contains_term(
            query_lower,
            term
        )
        for term in causality_terms
    ):

        mandatory_kbs.append(
            "KB-010"
        )


    # SHAP / explainability guardrail
    shap_terms = [
        "shap",
        "explainability",
        "feature importance",
        "model contribution"
    ]

    if any(
        contains_term(
            query_lower,
            term
        )
        for term in shap_terms
    ):

        mandatory_kbs.append(
            "KB-008"
        )


    # Batch disposition guardrail
    disposition_patterns = [
        r"\breject(?:\s+the)?\s+batch\b",
        r"\brelease(?:\s+the)?\s+batch\b",
        r"\bapprove(?:\s+the)?\s+batch\b",
        r"\bbatch disposition\b"
    ]

    if any(
        re.search(
            pattern,
            query_lower
        )
        for pattern in disposition_patterns
    ):

        mandatory_kbs.append(
            "KB-012"
        )


    # ----------------------------------------
    # Build final retrieval list
    # ----------------------------------------

    final_results = []


    # First guarantee mandatory evidence
    for kb_id in mandatory_kbs:

        result = next(
            (
                item
                for item in results
                if item["kb_id"] == kb_id
            ),
            None
        )

        if result is not None:

            final_results.append(
                result
            )


    # Then add highest-ranked results
    for result in results:

        if result["kb_id"] in [
            item["kb_id"]
            for item in final_results
        ]:
            continue


        final_results.append(
            result
        )


        if len(final_results) >= top_k:
            break


    return final_results[:top_k]


# ============================================
# TEST QUERIES
# ============================================

if __name__ == "__main__":

    test_queries = [

        (
            "What should I investigate "
            "when batch temperature is "
            "unusually high?"
        ),

        (
            "Can SHAP prove that temperature "
            "caused the batch failure?"
        ),

        (
            "What should I check if raw "
            "material purity is abnormal?"
        ),

        (
            "Should I reject the batch "
            "because the model predicts "
            "poor quality?"
        )
    ]


    for query in test_queries:

        print("\n")
        print("=" * 85)

        print(
            "QUERY:",
            query
        )

        print("=" * 85)


        results = retrieve_knowledge(
            query=query,
            top_k=3
        )


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRANK {rank}"
            )

            print(
                "KB:",
                result["kb_id"]
            )

            print(
                "TITLE:",
                result["title"]
            )

            print(
                "SEMANTIC:",
                round(
                    result["semantic_score"],
                    4
                )
            )

            print(
                "INTENT MATCH:",
                result["intent_match"]
            )

            print(
                "MATCHED KEYWORDS:",
                result["matched_keywords"]
            )

            print(
                "HYBRID SCORE:",
                round(
                    result["hybrid_score"],
                    4
                )
            )