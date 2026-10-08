
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from planner import create_research_plan
from web_search import search_web
from writer import write_answer


# ==================================================
# 1. CONFIGURATION
# ==================================================

MAX_SEARCH_ATTEMPTS = 3
MIN_SOURCES = 3


# ==================================================
# 2. STATE : données partagées
# ==================================================

class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    search_results: list[dict[str, str]]
    search_attempts: int
    final_answer: str


# ==================================================
# 3. PLANNER NODE
# ==================================================

def planner_node(state: ResearchState) -> dict:
    """
    Generate research sub-questions using Groq.
    """

    plan = create_research_plan(state["question"])

    print(f"\nPlanner : {len(plan.sub_questions)} sous-questions")

    return {
        "sub_questions": plan.sub_questions
    }


# ==================================================
# 4. SEARCH NODE
# ==================================================

def search_node(state: ResearchState) -> dict:
    """
    Search the web and accumulate unique sources.
    """

    # Numéro de la nouvelle tentative
    attempt = state["search_attempts"] + 1

    print(
        f"\n=== SEARCH ATTEMPT "
        f"{attempt}/{MAX_SEARCH_ATTEMPTS} ==="
    )

    # Récupérer les sous-questions du Planner
    questions = state["sub_questions"]

    # Sécurité : si aucune sous-question n'existe
    if not questions:
        questions = [state["question"]]

    # Choisir une sous-question différente
    question_index = (attempt - 1) % len(questions)
    query = questions[question_index]

    print(f"Recherche : {query}")

    # Appel réel de Tavily
    new_sources = search_web(query)

    # Récupérer les sources précédentes
    results = state["search_results"].copy()

    # URLs déjà collectées
    seen_urls = {
        source["url"]
        for source in results
    }

    # Vérifier les nouveaux résultats
    for source in new_sources:

        title = source["title"].lower()

        # Filtre temporaire pour notre exemple
        # RAG vs Fine-tuning
        has_rag = "rag" in title

        has_finetuning = (
            "fine-tun" in title
            or "fine tun" in title
        )

        # Éviter les sources hors sujet
        if not (has_rag and has_finetuning):
            continue

        # Éviter les doublons
        url = source["url"]

        if url not in seen_urls:
            results.append(source)
            seen_urls.add(url)

    print(f"Sources cumulées : {len(results)}")

    # Mettre à jour le State
    return {
        "search_results": results,
        "search_attempts": attempt
    }


# ==================================================
# 5. SUPERVISOR : ROUTING
# ==================================================

def supervisor_route(state: ResearchState) -> str:
    """
    Decide whether to search again, write,
    or stop the research.
    """

    source_count = len(state["search_results"])
    attempts = state["search_attempts"]

    # Cas 1 : assez de sources
    if source_count >= MIN_SOURCES:

        print("\nSupervisor : assez de sources -> Writer")
        return "writer"

    # Cas 2 : sources insuffisantes
    # et tentatives encore disponibles
    if attempts < MAX_SEARCH_ATTEMPTS:

        print("\nSupervisor : recherche supplémentaire")
        return "search"

    # Cas 3 : limite atteinte, quelques sources
    if source_count > 0:

        print("\nSupervisor : limite atteinte -> Writer")
        return "writer"

    # Cas 4 : limite atteinte, aucune source
    print("\nSupervisor : aucune source -> Stop")
    return "no_sources"


# ==================================================
# 6. WRITER NODE
# ==================================================

def writer_node(state: ResearchState) -> dict:
    """
    Generate a grounded answer using web evidence.
    """

    question = state["question"]
    sources = state["search_results"]

    print("\nWriter : génération de la réponse...")

    answer = write_answer(question, sources)

    # Signaler les preuves limitées
    if len(sources) < MIN_SOURCES:
        answer = (
            "ATTENTION : réponse basée sur un nombre "
            "limité de sources.\n\n"
            + answer
        )

    return {
        "final_answer": answer
    }


# ==================================================
# 7. NO SOURCES NODE
# ==================================================

def no_sources_node(state: ResearchState) -> dict:
    """
    Stop when no usable source was collected.
    """

    return {
        "final_answer": (
            "Recherche interrompue : aucune source "
            "pertinente n'a été trouvée après "
            f"{state['search_attempts']} tentative(s)."
        )
    }


# ==================================================
# 8. CONSTRUCTION LANGGRAPH
# ==================================================

builder = StateGraph(ResearchState)

# Enregistrer les Nodes
builder.add_node("planner", planner_node)
builder.add_node("search", search_node)
builder.add_node("writer", writer_node)
builder.add_node("no_sources", no_sources_node)

# Début du workflow
builder.add_edge(START, "planner")

# Après Planner, aller à Search
builder.add_edge("planner", "search")

# Supervisor : choisir la prochaine étape
builder.add_conditional_edges(
    "search",
    supervisor_route,
    {
        "search": "search",
        "writer": "writer",
        "no_sources": "no_sources"
    }
)

# Fin du workflow
builder.add_edge("writer", END)
builder.add_edge("no_sources", END)

# Compiler
graph = builder.compile()


# ==================================================
# 9. EXÉCUTION
# ==================================================

if __name__ == "__main__":

    initial_state = {
        "question": "Compare RAG et Fine-tuning",
        "sub_questions": [],
        "search_results": [],
        "search_attempts": 0,
        "final_answer": ""
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 15}
    )

    print("\n=== RESEARCH PLAN ===")

    for i, question in enumerate(
        result["sub_questions"], start=1
    ):
        print(f"{i}. {question}")

    print("\n=== SOURCES ===")

    for source in result["search_results"]:
        print(f"Title: {source['title']}")
        print(f"URL: {source['url']}")
        print("---")

    print("\n=== RESEARCH STATS ===")
    print("Tentatives :", result["search_attempts"])
    print("Sources :", len(result["search_results"]))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
