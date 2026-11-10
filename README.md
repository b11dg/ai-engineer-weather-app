# Weather Assistant

Python script that gets current weather for a city using Llama 3.1 via OpenRouter API.

## Prerequisites

- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai)

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd ai-engineer-course

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure API key
cp .env.example .env
# Edit .env and add your OpenRouter API key

# 6. Run
python weather.py
```

## Usage

```
$ python weather.py
In which city are you interested in the weather? London
The current weather in London is:
* Temperature: 12°C (54°F)
* Conditions: Overcast with light rain showers
...
```

## Project Structure

```
.
├── weather.py          # Main script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your API keys (not committed)
├── .gitignore          # Git ignore rules
└── venv/               # Virtual environment (not committed)
```

## Dependencies

- `requests` - HTTP library for API calls
- `python-dotenv` - Load environment variables from .env file

## License

MIT