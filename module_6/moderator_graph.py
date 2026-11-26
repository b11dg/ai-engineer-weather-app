"""
Comment moderation graph (module 6): a LangGraph state machine.

The graph reads a blog comment and sends it to the right handler:

    START -> analyze_comment -> (TOXIC)   -> toxic_handler -> END
                             -> (QUESTION) -> qa_handler    -> END
                             -> (NORMAL)   -> END

analyze_comment asks the model for exactly one word (TOXIC, QUESTION or
NORMAL) and stores it in state["category"]. A conditional edge picks the
next node from that category: toxic_handler deletes the comment with a
fixed reply, qa_handler asks the model for a polite answer, and NORMAL
goes straight to END.

Run:
    python module_6/moderator_graph.py

API:    OpenRouter via langchain-openai (key in ../.env as MODEL_API_KEY)
Model:  NVIDIA Nemotron 3.5 Super VL, free tier
"""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3.5-super-vl:free"

if not MODEL_API_KEY:
    raise SystemExit("MODEL_API_KEY is not set. Put it in the .env file in the repo root.")

# The analyzer needs the same answer every time, the answer writer needs
# some variety, so the two prompts use two model settings.
classifier_model = ChatOpenAI(
    model=MODEL,
    api_key=MODEL_API_KEY,
    base_url=BASE_URL,
    temperature=0,
)
answer_model = ChatOpenAI(
    model=MODEL,
    api_key=MODEL_API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
)

CATEGORIES = ("TOXIC", "QUESTION", "NORMAL")


class AgentState(TypedDict):
    comment: str
    category: str
    final_response: str


# --- Node 1: analyzer -------------------------------------------------------

analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ти модератор коментарів блогу. Відповідай лише одним словом."),
    ("human",
     "Визнач категорію коментаря:\n"
     "- TOXIC — образи, агресія або приниження;\n"
     "- QUESTION — питання по темі статті;\n"
     "- NORMAL — звичайний відгук.\n"
     "Коментар: {comment}\n"
     "Виведи лише одне слово: TOXIC, QUESTION або NORMAL."),
])
analyzer_chain = analyzer_prompt | classifier_model | StrOutputParser()


def analyze_comment(state: AgentState) -> dict:
    """Ask the model for one category word and store it in the state."""
    answer = analyzer_chain.invoke({"comment": state["comment"]})
    category = answer.strip().upper()

    # The model may wrap the word in punctuation or a short sentence, so also
    # look for the label inside a longer answer.
    if category not in CATEGORIES:
        category = next((label for label in CATEGORIES if label in category), "")

    if category not in CATEGORIES:
        raise ValueError(f"Unexpected category from the model: {answer!r}")

    return {"category": category}


# --- Node 2: handlers -------------------------------------------------------

TOXIC_RESPONSE = "Ваш коментар видалено через порушення правил."

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ти ввічливий консультант блогу. Відповідай коротко і по суті, "
     "простим текстом без Markdown."),
    ("human",
     "Дай ввічливу відповідь на питання читача тією самою мовою, "
     "якою воно написане. 2-4 речення, без вітань і підпису:\n{comment}"),
])
qa_chain = qa_prompt | answer_model | StrOutputParser()


def toxic_handler(state: AgentState) -> dict:
    """Delete a toxic comment with a fixed reply."""
    return {"final_response": TOXIC_RESPONSE}


def qa_handler(state: AgentState) -> dict:
    """Ask the model for a polite answer to the reader's question."""
    answer = qa_chain.invoke({"comment": state["comment"]})
    return {"final_response": answer.strip()}


# --- Conditional edge: router ----------------------------------------------

def route_by_category(state: AgentState) -> str:
    """Pick the next node from the category found by the analyzer."""
    if state["category"] == "TOXIC":
        return "toxic_handler"
    if state["category"] == "QUESTION":
        return "qa_handler"
    return END


# --- Graph -----------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("analyze_comment", analyze_comment)
builder.add_node("toxic_handler", toxic_handler)
builder.add_node("qa_handler", qa_handler)

builder.add_edge(START, "analyze_comment")
builder.add_conditional_edges(
    "analyze_comment",
    route_by_category,
    {
        "toxic_handler": "toxic_handler",
        "qa_handler": "qa_handler",
        END: END,
    },
)
builder.add_edge("toxic_handler", END)
builder.add_edge("qa_handler", END)

graph = builder.compile()


COMMENTS = [
    "Ця стаття повний відстій, автор дурень!",
    "А як встановити цю бібліотеку на Windows?",
    "Дякую, дуже цікаво!",
]


def print_state(state: AgentState) -> None:
    """Print one final state in a readable form."""
    response = state["final_response"].strip() or "— (відповідь не потрібна)"
    print(f"Коментар:  {state['comment']}")
    print(f"Категорія: {state['category']}")
    print(f"Відповідь: {response}")


def main() -> None:
    for index, comment in enumerate(COMMENTS, start=1):
        print(f"\n=== Запуск {index} ===")
        final_state = graph.invoke(
            {"comment": comment, "category": "", "final_response": ""}
        )
        print_state(final_state)


if __name__ == "__main__":
    main()
