
import os
from dotenv import load_dotenv
from groq import Groq
from schemas import ResearchPlan

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def create_research_plan(question: str) -> ResearchPlan:

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. "
                    "Analyze the user question and divide it "
                    "into useful research sub-questions. "
                    "Set a priority from 1 (highest) to 5 (lowest). "
                    "Respond in French."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "research_plan",
                "strict": True,
                "schema": ResearchPlan.model_json_schema()
            }
        },
        reasoning_effort="low",
        max_completion_tokens=1000
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Le LLM n'a retourné aucun plan.")

    return ResearchPlan.model_validate_json(content)


if __name__ == "__main__":

    question = "Compare RAG et Fine-tuning"

    plan = create_research_plan(question)

    print("=== RESEARCH PLAN ===")
    print(plan.model_dump_json(indent=2))
