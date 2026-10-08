
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from planner import create_research_plan


# 1. Définition du State partagé
class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    summary: str


# 2. Premier Node : Planner
def planner_node(state: ResearchState) -> dict:

    question = state["question"]

    plan = create_research_plan(question)

    return {
        "sub_questions": plan.sub_questions
    }


# 3. Deuxième Node : Formatter
def formatter_node(state: ResearchState) -> dict:

    questions = state["sub_questions"]

    summary = "\n".join(
        f"{i}. {q}"
        for i, q in enumerate(questions, start=1)
    )

    return {"summary": summary}


# 4. Construire notre graphe
builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("formatter", formatter_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "formatter")
builder.add_edge("formatter", END)

# 5. Compiler
graph = builder.compile()


# 6. Exécuter
if __name__ == "__main__":

    result = graph.invoke({
        "question": "Compare RAG et Fine-tuning",
        "sub_questions": [],
        "summary": ""
    })

    print("=== RESEARCH PLAN ===")
    print(result["summary"])
