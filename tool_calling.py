
import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# 1. Notre première fonction Python utilisable comme outil
def search_local_notes(query: str) -> str:

    documents = {
        "rag": (
            "RAG récupère des informations depuis une "
            "source externe avant de générer une réponse."
        ),
        "fine-tuning": (
            "Le fine-tuning poursuit l'entraînement "
            "d'un modèle pour adapter son comportement."
        )
    }

    results = [
        text
        for keyword, text in documents.items()
        if keyword in query.lower()
    ]

    if not results:
        return "Aucune information trouvée."

    return "\n".join(results)


# 2. Décrire l'outil au LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_local_notes",
            "description": (
                "Search local notes about RAG and fine-tuning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topics to search for"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]



# 3. Paramètres de notre agent
MAX_ITERATIONS = 4


def run_agent(question: str) -> str:

    # Historique de la conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. "
                "Use search_local_notes to find evidence. "
                "For comparisons, retrieve information "
                "about both topics before answering. "
                "Do not invent facts missing from the notes. "
                "Answer in French."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Boucle de décision
    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n=== ITERATION {iteration} ===")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            reasoning_effort="low",
            max_completion_tokens=1024
        )

        assistant_message = response.choices[0].message

        # Conserver la décision du LLM
        messages.append(assistant_message)

        # Si aucun outil n'est demandé, l'agent termine
        if not assistant_message.tool_calls:

            print("Agent : aucune autre recherche demandée.")

            return assistant_message.content or (
                "Aucune réponse finale générée."
            )

        # Exécuter les outils demandés
        for tool_call in assistant_message.tool_calls:

            function_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                if function_name != "search_local_notes":
                    raise ValueError("Outil non autorisé")

                query = arguments["query"]

                if not isinstance(query, str):
                    raise ValueError("Query doit être un texte")

                result = search_local_notes(query)

            except (ValueError, KeyError, TypeError) as error:
                result = f"Erreur d'outil : {error}"

            print("Outil :", function_name)
            print("Résultat :", result)

            # Retourner le résultat au LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return (
        "Recherche arrêtée : nombre maximal "
        "d'itérations atteint."
    )


if __name__ == "__main__":

    question = (
        "Compare RAG et fine-tuning. "
        "Consulte les notes sur les deux sujets "
        "avant de répondre."
    )

    final_answer = run_agent(question)

    print("\n=== REPONSE FINALE ===")
    print(final_answer)


# 4. Premier appel : le LLM demande un outil
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="required",
    max_completion_tokens=1024
)

assistant_message = response.choices[0].message

messages.append(assistant_message)


# 5. Python exécute les outils demandés
for tool_call in assistant_message.tool_calls or []:

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print("Outil demandé :", function_name)
    print("Arguments :", arguments)

    if function_name == "search_local_notes":
        result = search_local_notes(arguments["query"])
    else:
        result = "Outil inconnu."

    print("Résultat :", result)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })


# 6. Deuxième appel : le LLM reçoit les résultats
final_response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_completion_tokens=1024
)

print("\n=== RÉPONSE FINALE ===")
print(final_response.choices[0].message.content)
