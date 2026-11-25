"""
Content pipeline (module 5): two LangChain chains joined with LCEL.

Chain 1 (idea generator):  {topic} -> one catchy article title
Chain 2 (writer):          {title} -> a short intro paragraph

overall_chain runs chain 1, passes its output to chain 2 as {title}
and returns the topic, the title and the intro together.

API:    OpenRouter via langchain-openai (key in ../.env as MODEL_API_KEY)
Model:  NVIDIA Nemotron 3.5 Super VL, free tier
"""

import os
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


def main() -> None:
    result = overall_chain.invoke({"topic": "Майбутнє космічного туризму"})

    print(f"Тема: {result['topic']}\n")
    print(result["title"].strip())
    print()
    print(result["intro"].strip())


if __name__ == "__main__":
    main()
