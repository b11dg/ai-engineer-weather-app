# AI Engineer Course

Homework for the AI Engineer course, organized by module.

**Repository:** https://github.com/b11dg/ai-engineer-weather-app

## Modules

| Module | Project | Location |
|--------|---------|----------|
| 1 | Weather Assistant: real current weather for a city, described by an LLM | [`module_1/`](module_1/README.md) |
| 2 | Temperature Arena: same prompt at temperature 0 and 1, compared | [`module_2/`](module_2/README.md) |

Both modules use free OpenRouter models (`:free` suffix), so no credits are charged. All modules share one virtual environment, one `requirements.txt` and one `.env` in the repo root.

## Setup

- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai)

```bash
# 1. Clone the repository
git clone https://github.com/b11dg/ai-engineer-weather-app.git
cd ai-engineer-weather-app

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set MODEL_API_KEY to your OpenRouter API key
```

---

## Module 1: Weather Assistant

Gets real current weather for a city from [Open-Meteo](https://open-meteo.com) (free, no key) and has NVIDIA Nemotron 3.5 Lightning (free tier, `nvidia/nemotron-3.5-lightning:free`) turn it into a short, friendly answer via the OpenRouter API. Details (in Russian) are in [`module_1/README.md`](module_1/README.md).

```
$ python module_1/weather.py
In which city are you interested in the weather? London
At 7°C in London, it feels more like 3.6°C with a clear sky and 52% humidity. ...
```

## Module 2: Temperature Arena

Sends the same prompt to NVIDIA Nemotron 3 Ultra (free tier, `nvidia/nemotron-3-ultra-550b-a55b:free`) at temperature 0.0 and 1.0 and prints both answers for comparison. Details, observations and conclusions (in Russian) are in [`module_2/README.md`](module_2/README.md).

```bash
python module_2/arena.py
```

## Project Structure

```
.
├── module_1/
│   ├── weather.py      # Module 1: weather assistant
│   └── README.md       # Module 1 description
├── module_2/
│   ├── arena.py        # Module 2: temperature arena
│   └── README.md       # Module 2 description and findings
├── requirements.txt    # Python dependencies (shared)
├── .env.example        # Environment variables template
├── .env                # Your API key (not committed)
├── .gitignore          # Git ignore rules
└── venv/               # Virtual environment (not committed)
```

## Dependencies

- `requests` - HTTP library for API calls (OpenRouter, Open-Meteo)
- `python-dotenv` - Load environment variables from .env file

## License

MIT
