# AI Engineer Course

Homework for the AI Engineer course, organized by module.

**Repository:** https://github.com/b11dg/ai-engineer-weather-app

## Modules

| Module | Project | Location |
|--------|---------|----------|
| 1 | Weather Assistant: real current weather for a city, described by an LLM | [`module_1/`](module_1/README.md) |
| 2 | Temperature Arena: same prompt at temperature 0 and 1, compared | [`module_2/`](module_2/README.md) |
| 3 | Support Message Extractor: customer message to a structured JSON ticket | [`module_3/`](module_3/README.md) |
| 4 | Smart Inventory Agent: Function Calling with a mock warehouse, in a desktop GUI | [`module_4/`](module_4/README.md) |

All modules use free OpenRouter models (`:free` suffix), so no credits are charged. All modules share one virtual environment, one `requirements.txt` and one `.env` in the repo root.

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

## Module 3: Support Message Extractor

A customer types a support message in the terminal. NVIDIA Nemotron 3.5 Super VL (free tier, `nvidia/nemotron-3.5-super-vl:free`) reads it with a System Prompt and, in JSON Mode, returns a structured ticket: name, email, order ID, category, sentiment, urgency, summary, requested actions and a draft reply. The script parses the JSON, prints it nicely, handles invalid JSON with try/except, and appends every ticket to `module_3/data.json` (created on the first run). It also guards against prompt injection and limits the message and the answer to 4000 characters each. It calls OpenRouter through the `openai` library. Details (in Russian) are in [`module_3/README.md`](module_3/README.md).

```bash
python module_3/extractor.py
```

## Module 4: Smart Inventory Agent

A desktop chat window (PyQt6, Windows/macOS/Linux) where a customer asks whether products are in stock. NVIDIA Nemotron 3.5 Super VL (free tier, `nvidia/nemotron-3.5-super-vl:free`) gets a `check_inventory` tool whose JSON Schema is generated from a Pydantic model (`"strict": true`, `"additionalProperties": false`). The app intercepts the model's `tool_calls`, validates the arguments with the same Pydantic model, runs a local mock warehouse function, sends the results back as `tool` messages and streams the final answer (`stream=True`) into the window. An agent log panel shows every step of the loop. Details (in Russian) are in [`module_4/README.md`](module_4/README.md).

```bash
python module_4/smart_agent.py        # GUI
python module_4/smart_agent.py --cli  # terminal, no window
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
├── module_3/
│   ├── extractor.py    # Module 3: support message to JSON ticket
│   ├── README.md       # Module 3 description
│   └── docs/plan.md    # Module 3 implementation plan
├── module_4/
│   ├── smart_agent.py  # Module 4: Function Calling agent with a GUI
│   └── README.md       # Module 4 description
├── requirements.txt    # Python dependencies (shared)
├── .env.example        # Environment variables template
├── .env                # Your API key (not committed)
├── .gitignore          # Git ignore rules
└── venv/               # Virtual environment (not committed)
```

## Dependencies

- `requests` - HTTP library for API calls (OpenRouter, Open-Meteo)
- `python-dotenv` - Load environment variables from .env file
- `openai` - OpenAI-compatible client, used with OpenRouter in modules 3 and 4
- `pydantic` - data models; module 4 builds the tool's JSON Schema from one and validates tool arguments
- `PyQt6` - cross-platform GUI for module 4 (pinned to 6.9, the last Qt line that runs on macOS 12)

## License

MIT
