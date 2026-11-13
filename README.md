# AI Engineer Course

Homework for the AI Engineer course, organized by module.

**Repository:** https://github.com/b11dg/ai-engineer-weather-app

## Modules

| Module | Project | Location |
|--------|---------|----------|
| 1 | Weather Assistant: current weather for a city via an LLM | [`weather.py`](weather.py) (repo root) |
| 2 | Temperature Arena: same prompt at temperature 0 and 1, compared | [`module_2/`](module_2/README.md) |

All modules share one virtual environment, one `requirements.txt` and one `.env` in the repo root.

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

Python script that gets current weather for a city using Llama 3.1 via OpenRouter API.

### Usage

```
$ python weather.py
In which city are you interested in the weather? London
The current weather in London is:
* Temperature: 12°C (54°F)
* Conditions: Overcast with light rain showers
...
```

## Module 2: Temperature Arena

Sends the same prompt to NVIDIA Nemotron 3 Ultra at temperature 0.0 and 1.0 and prints both answers for comparison. Details, observations and conclusions (in Russian) are in [`module_2/README.md`](module_2/README.md).

```bash
python module_2/arena.py
```

## Project Structure

```
.
├── weather.py          # Module 1: weather assistant
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

- `requests` - HTTP library for API calls
- `python-dotenv` - Load environment variables from .env file

## License

MIT
