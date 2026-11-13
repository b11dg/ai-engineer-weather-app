"""
Temperature arena: send the same prompt to the same model twice,
once with temperature 0 and once with a high temperature, and compare.

API:    OpenRouter (key in ../.env as MODEL_API_KEY)
Model:  NVIDIA Nemotron 3 Ultra
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

TEMPERATURES = [0.0, 1.0]

# A medium-difficulty prompt: it mixes a creative part (name, slogan, drinks),
# where temperature should change the result a lot, with a fixed structure,
# so the two answers are easy to compare side by side.
PROMPT = (
    "Придумай кофейню, которая находится прямо внутри IT-офиса.\n"
    "Дай ответ строго в таком формате:\n"
    "1. Название (одно)\n"
    "2. Слоган (одна фраза)\n"
    "3. Описание атмосферы (3-4 предложения)\n"
    "4. Три необычных напитка для меню: название и одно предложение, "
    "почему он понравится программистам."
)


def ask(prompt: str, temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {MODEL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        # Nemotron 3 Ultra reasons before answering, and reasoning tokens
        # count toward this limit, so it has to be generous.
        "max_tokens": 3000,
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except (KeyError, IndexError):
        return "Error: unexpected response format"


def main():
    print(f"Model: {MODEL}")
    print(f"Prompt:\n{PROMPT}\n")

    for temperature in TEMPERATURES:
        print("=" * 70)
        print(f"TEMPERATURE = {temperature}")
        print("=" * 70)
        print(ask(PROMPT, temperature))
        print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# НАБЛЮДЕНИЯ (скрипт запускался 2 раза, Nemotron 3 Ultra через OpenRouter)
# ---------------------------------------------------------------------------
#
# Temperature = 0.0
#   - Текст чистый и грамотный: русский язык без ошибок, формат 1-2-3-4
#     соблюден точно, описание укладывается в просимые 3-4 предложения.
#   - Идеи "самые ожидаемые": Runtime / Root Access, Garbage Collector,
#     Hot Reload, Legacy Latte, резиновые уточки, механические клавиатуры.
#   - Неожиданно: ответ НЕ повторяется между запусками (в первый раз
#     "Runtime", во второй "Root Access"). Температура 0 не гарантирует
#     детерминизм: OpenRouter может отправить запрос разным провайдерам,
#     на GPU с батчингом есть мелкий численный шум, а модель сначала
#     рассуждает (reasoning), и небольшое расхождение в рассуждении
#     уводит весь ответ в другую сторону.
#
# Temperature = 1.0
#   - Ответы длиннее и "цветистее": больше деталей и шуток (виски в
#     латте "для смелости деплоить в пятницу", Wi-Fi 10 Гбит/с,
#     свет по "circadian rhythm API").
#   - Ядро идей во многом то же (Runtime, Garbage Collector, Legacy Latte),
#     меняются формулировки и детали. Самые вероятные идеи модели
#     остаются сверху даже при высокой температуре.
#   - Главный минус: появляются "глюки" при выборе маловероятных токенов,
#     а именно смешение языков и опечатки: "рестриretto", "мониторными
#     braccio", "кoldt brew", "капелью чили" и даже корейское "맛" посреди
#     русской фразы. При температуре 0 такого не было ни разу.
#   - Формат все равно соблюден, но описание вылезает за 3-4 предложения.
#
# Вывод
#   - Низкая температура подходит, когда важны точность, чистый язык и
#     следование инструкции (факты, код, извлечение данных, JSON).
#   - Высокая температура дает больше разнообразия и неожиданных деталей
#     (брейншторм, креатив), но ценой качества текста: ответ надо
#     проверять. Для креатива разумный компромисс где-то 0.6-0.8.
#   - Полную повторяемость не дает даже temperature 0, если нужна
#     воспроизводимость, нужен seed и фиксированный провайдер.
