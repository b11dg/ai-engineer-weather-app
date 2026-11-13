"""
Weather assistant (module 1).

1. Open-Meteo finds the city and returns its real current weather
   (free, no API key).
2. A free LLM on OpenRouter turns that data into a short, friendly answer.

API:    OpenRouter (key in ../.env as MODEL_API_KEY), Open-Meteo
Model:  NVIDIA Nemotron 3.5 Lightning, free tier (no credits charged)
"""

import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free OpenRouter model (no credits charged)
MODEL = "nvidia/nemotron-3.5-lightning:free"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

NOT_FOUND = "Sorry, choose another city"

# WMO weather interpretation codes used by Open-Meteo
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

SYSTEM_PROMPT = (
    "You are a weather assistant. You receive real current weather data for a city. "
    "Write a short, friendly answer (3-5 sentences) that covers the temperature, "
    "how it feels, the conditions, humidity and wind, and end with one practical tip "
    "(what to wear or take along). Use only the data provided, never invent numbers."
)


def find_city(city: str) -> Optional[dict]:
    """Return the best Open-Meteo geocoding match for the city, or None."""
    response = requests.get(GEOCODING_URL, params={"name": city, "count": 1}, timeout=20)
    response.raise_for_status()
    results = response.json().get("results")
    return results[0] if results else None


def fetch_current_weather(latitude: float, longitude: float) -> dict:
    """Return Open-Meteo current weather values and their units."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                   "weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return {"values": data["current"], "units": data["current_units"]}


def format_weather_data(place: dict, weather: dict) -> str:
    """Turn the raw API data into plain text the LLM can read."""
    v, u = weather["values"], weather["units"]
    location = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    conditions = WEATHER_CODES.get(v["weather_code"], f"weather code {v['weather_code']}")
    return (
        f"City: {location}\n"
        f"Local time: {v['time']}\n"
        f"Temperature: {v['temperature_2m']}{u['temperature_2m']}\n"
        f"Feels like: {v['apparent_temperature']}{u['apparent_temperature']}\n"
        f"Conditions: {conditions}\n"
        f"Humidity: {v['relative_humidity_2m']}{u['relative_humidity_2m']}\n"
        f"Wind: {v['wind_speed_10m']} {u['wind_speed_10m']}"
    )


def ask_llm(weather_text: str) -> str:
    """Ask the free OpenRouter model to describe the weather data."""
    headers = {
        "Authorization": f"Bearer {MODEL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": weather_text},
        ],
        "temperature": 0.3,
        # The model reasons before answering and reasoning tokens count
        # toward this limit, so it has to be generous.
        "max_tokens": 2000,
    }
    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return (content or "").strip()


def get_weather(city: str) -> str:
    try:
        place = find_city(city)
        if place is None:
            return NOT_FOUND
        weather = fetch_current_weather(place["latitude"], place["longitude"])
    except requests.exceptions.RequestException as e:
        return f"Error: weather service is unavailable ({e})"
    except (KeyError, ValueError):
        return "Error: unexpected response from the weather service"

    weather_text = format_weather_data(place, weather)

    try:
        answer = ask_llm(weather_text)
    except requests.exceptions.RequestException as e:
        return f"Error: language model is unavailable ({e})\n\nRaw data:\n{weather_text}"
    except (KeyError, IndexError, ValueError):
        return f"Error: unexpected response from the language model\n\nRaw data:\n{weather_text}"

    return answer or f"Error: the language model returned an empty answer\n\nRaw data:\n{weather_text}"


def main():
    city = input("In which city are you interested in the weather? ").strip()

    if not city:
        print("Please enter a valid city name.")
        return

    print(get_weather(city))


if __name__ == "__main__":
    main()
