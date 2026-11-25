"""
Content pipeline (module 5): two LangChain chains joined with LCEL.

Chain 1 (idea generator):  {topic} -> one catchy article title
Chain 2 (writer):          {title} -> a short intro paragraph

overall_chain runs chain 1, passes its output to chain 2 as {title}
and returns the topic, the title and the intro together.

Usage:
    python module_5/content_pipeline.py           # invoke(): print when done
    python module_5/content_pipeline.py --stream  # stream(): print as generated

API:    OpenRouter via langchain-openai (key in ../.env as MODEL_API_KEY)
Model:  NVIDIA Nemotron 3.5 Super VL, free tier
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3.5-super-vl:free"

if not MODEL_API_KEY:
    raise SystemExit("MODEL_API_KEY is not set. Put it in the .env file in the repo root.")

# OpenRouter speaks the OpenAI API, so ChatOpenAI works with it as is.
model = ChatOpenAI(
    model=MODEL,
    api_key=MODEL_API_KEY,
    base_url=BASE_URL,
    temperature=0.8,
)

# --- Chain 1: idea generator ------------------------------------------------
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "Ти креативний редактор онлайн-журналу. Відповідай українською."),
    ("human",
     "Придумай 1 яскравий заголовок для статті на тему: {topic}.\n"
     "Виведи лише сам заголовок, без лапок, пояснень і варіантів."),
])
chain1 = prompt1 | model | StrOutputParser()

# --- Chain 2: writer --------------------------------------------------------
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "Ти досвідчений автор науково-популярних статей. Відповідай українською."),
    ("human",
     "Напиши короткий абзац-вступ (3-5 речень) для статті із заголовком:\n"
     "{title}\n"
     "Виведи лише текст абзацу, без заголовка."),
])
chain2 = prompt2 | model | StrOutputParser()

# --- Overall pipeline (LCEL) ------------------------------------------------
# RunnablePassthrough.assign() keeps the incoming dict and adds a new key:
#   {"topic"} -> {"topic", "title"} -> {"topic", "title", "intro"}
# The dict that reaches chain 2 has a "title" key, so it fills {title}.
# A shorter form, {"title": chain1} | chain2, would return only the intro.
overall_chain = (
    RunnablePassthrough.assign(title=chain1)
    | RunnablePassthrough.assign(intro=chain2)
)


TOPIC = "Майбутнє космічного туризму"


def run_invoke() -> None:
    """Wait for the whole pipeline, then print the result."""
    result = overall_chain.invoke({"topic": TOPIC})

    print(f"Тема: {result['topic']}\n")
    print(result["title"].strip())
    print()
    print(result["intro"].strip())


def run_stream() -> None:
    """Print the title and the intro piece by piece as the model writes them.

    stream() yields small dicts with one key each, in order:
        {"topic": "..."}                  the input, at once
        {"title": "Кос"}, {"title": ...}  chain 1 output, piece by piece
        {"intro": "У"},   {"intro": ...}  chain 2 output, piece by piece
    Chain 2 starts only after the whole title is ready, because its prompt
    needs the full {title}.
    """
    current_key = None
    for chunk in overall_chain.stream({"topic": TOPIC}):
        for key, text in chunk.items():
            if key != current_key:  # a new section: topic -> title -> intro
                if current_key is not None:
                    print("\n")
                current_key = key
                if key == "topic":
                    print("Тема: ", end="")
            print(text, end="", flush=True)  # flush shows each piece at once
    print()


def main() -> None:
    if "--stream" in sys.argv[1:]:
        run_stream()
    else:
        run_invoke()


if __name__ == "__main__":
    main()
