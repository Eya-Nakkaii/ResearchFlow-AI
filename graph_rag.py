
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from vector_store import search_documents
from rag_answer import write_answer_from_matches


# ==========================================
# 1. STATE
# ==========================================

class RAGState(TypedDict):

    question: str
    matches: list[dict]
    final_answer: str


# ==========================================
# 2. RETRIEVER NODE
# ==========================================

def retriever_node(state: RAGState) -> dict:

    print("\nRetriever : recherche dans ChromaDB...")

    question = state["question"]

    matches = search_documents(question)

    print(f"Passages trouvés : {len(matches)}")

    return {
        "matches": matches
    }


# ==========================================
# 3. SUPERVISOR
# ==========================================

def supervisor_route(state: RAGState) -> str:

    matches = state["matches"]

    if len(matches) > 0:

        print("Supervisor : passages trouvés -> Writer")

        return "writer"

    print("Supervisor : aucun passage -> Stop")

    return "no_sources"


# ==========================================
# 4. WRITER NODE
# ==========================================

def writer_node(state: RAGState) -> dict:

    print("\nWriter : génération de la réponse...")

    question = state["question"]

    matches = state["matches"]

    answer = write_answer_from_matches(
        question,
        matches
    )

    return {
        "final_answer": answer
    }


# ==========================================
# 5. NO SOURCES NODE
# ==========================================

def no_sources_node(state: RAGState) -> dict:

    return {
        "final_answer": (
            "Aucun passage documentaire trouvé. "
            "Impossible de générer une réponse sourcée."
        )
    }


# ==========================================
# 6. CONSTRUCTION DU GRAPHE
# ==========================================

builder = StateGraph(RAGState)

# Enregistrer les Nodes
builder.add_node("retriever", retriever_node)
builder.add_node("writer", writer_node)
builder.add_node("no_sources", no_sources_node)

# Première étape
builder.add_edge(START, "retriever")

# Choisir la suite selon les résultats
builder.add_conditional_edges(
    "retriever",
    supervisor_route,
    {
        "writer": "writer",
        "no_sources": "no_sources"
    }
)

# Terminer
builder.add_edge("writer", END)
builder.add_edge("no_sources", END)

# Compiler
graph = builder.compile()


# ==========================================
# 7. EXÉCUTION
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nQuestion sur le rapport industriel : "
    ).strip()

    if not question:
        print("Question vide.")
        raise SystemExit(1)

    initial_state = {
        "question": question,
        "matches": [],
        "final_answer": ""
    }

    result = graph.invoke(initial_state)

    print("\n=== RÉSULTAT FINAL ===\n")

    print(result["final_answer"])
