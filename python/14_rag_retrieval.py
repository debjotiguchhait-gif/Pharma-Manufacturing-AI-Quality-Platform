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

pattern = r"\[(KB-\d+)\]\s+([^\n]+)\n(.*?)(?=\n\[KB-\d+\]|\Z)"

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
# LOAD LOCAL EMBEDDING MODEL
# ============================================

print(
    "\nLoading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================
# CREATE KB EMBEDDINGS
# ============================================

kb_texts = [

    chunk["embedding_text"]

    for chunk in knowledge_chunks
]


print(
    "Creating knowledge embeddings..."
)


kb_embeddings = (
    embedding_model.encode(
        kb_texts,
        normalize_embeddings=True
    )
)


print(
    "Knowledge embeddings ready."
)


# ============================================
# RETRIEVAL FUNCTION
# ============================================

def retrieve_knowledge(
    query,
    top_k=3,
    minimum_similarity=0.20
):

    """
    Retrieve relevant knowledge-base
    sections using semantic similarity.
    """

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )


    similarities = cosine_similarity(
        query_embedding,
        kb_embeddings
    )[0]


    ranked_indices = np.argsort(
        similarities
    )[::-1]


    results = []


    for index in ranked_indices:

        score = float(
            similarities[index]
        )


        if score < minimum_similarity:
            continue


        chunk = knowledge_chunks[index]


        results.append({

            "kb_id":
                chunk["kb_id"],

            "title":
                chunk["title"],

            "similarity_score":
                round(score, 4),

            "content":
                chunk["content"]
        })


        if len(results) >= top_k:
            break


    return results


# ============================================
# TEST RETRIEVAL
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
        )

    ]


    for query in test_queries:

        print("\n")
        print("=" * 80)

        print(
            "QUERY:",
            query
        )

        print("=" * 80)


        retrieved = retrieve_knowledge(
            query=query,
            top_k=3
        )


        if not retrieved:

            print(
                "No sufficiently relevant "
                "knowledge retrieved."
            )

            continue


        for rank, result in enumerate(
            retrieved,
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
                "SIMILARITY:",
                result[
                    "similarity_score"
                ]
            )

            print(
                "\nCONTENT:"
            )

            print(
                result["content"]
            )