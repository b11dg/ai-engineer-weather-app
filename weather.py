import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_weather(city):
    headers = {
        "Authorization": f"Bearer {MODEL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a weather assistant. Provide current weather information for the requested city. If the city is not found or weather data is unavailable, respond with exactly: Sorry, choose another city"},
            {"role": "user", "content": f"What is the current weather in {city}?"}
        ],
        "temperature": 0.1,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except KeyError:
        return "Sorry, choose another city"

def main():
    city = input("In which city are you interested in the weather? ").strip()
    
    if not city:
        print("Please enter a valid city name.")
        return
    
    weather = get_weather(city)
    print(weather)

if __name__ == "__main__":
    main()