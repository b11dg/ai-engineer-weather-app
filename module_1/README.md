# Модуль 1. Погодный ассистент

> Это задание модуля 1: первое приложение, которое работает с языковой моделью через API. Задание модуля 2 лежит в [`module_2/`](../module_2/README.md). Обзор всех модулей есть в [корневом README](../README.md).

Пользователь вводит город, а скрипт отвечает коротким понятным текстом о текущей погоде там и даёт совет, что надеть.

- **LLM API:** [OpenRouter](https://openrouter.ai)
- **Модель:** NVIDIA Nemotron 3.5 Lightning, бесплатная версия (`nvidia/nemotron-3.5-lightning:free`)
- **Данные о погоде:** [Open-Meteo](https://open-meteo.com), бесплатно и без ключа
- **Файл:** [`weather.py`](weather.py)

## Как это работает

```
город ──► Open-Meteo Geocoding ──► координаты
                                      │
                                      ▼
          Open-Meteo Forecast ──► текущая погода (температура, ощущается как,
                                      │             влажность, ветер, код погоды)
                                      ▼
          OpenRouter (LLM)    ──► дружелюбный ответ + совет
```

1. **Поиск города.** Geocoding API Open-Meteo по названию возвращает координаты. Понимает разные языки, например «London» и «Санкт-Петербург». Если город не найден, скрипт отвечает `Sorry, choose another city`.
2. **Реальная погода.** Forecast API Open-Meteo возвращает текущие показатели. Числовой код погоды (WMO) скрипт переводит в текст, например `3 → overcast`.
3. **Ответ модели.** Данные отправляются в OpenRouter вместе с системным промптом. Модель пишет 3–5 предложений и добавляет один практический совет. В промпте прямо сказано брать только переданные данные и не придумывать числа.

### Зачем здесь отдельный погодный API

У языковой модели нет доступа к данным в реальном времени. Если просто спросить её о погоде, она либо выдумает прогноз (так делала Llama 3.1 8B, которая стояла здесь раньше), либо честно откажется отвечать (так делает Nemotron). Поэтому факты берутся из погодного API, а модель отвечает за то, что у неё получается лучше всего: превращает сухие цифры в понятный текст.

## Запуск

Окружение общее для всего репозитория, поэтому команды выполняются из корня проекта.

```bash
# 1. Виртуальное окружение и зависимости (если ещё не сделано)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Ключ OpenRouter в корневом .env
cp .env.example .env
# впишите ключ в MODEL_API_KEY=...

# 3. Запуск
python module_1/weather.py
```

Скрипт сам находит `.env` в корне репозитория, поэтому его можно запускать из любой папки. Ключ нужен только для OpenRouter, Open-Meteo работает без него.

## Пример

```
$ python module_1/weather.py
In which city are you interested in the weather? London
At 7°C in London, it feels more like 3.6°C with a clear sky and 52% humidity.
A gentle 6.8 km/h breeze is adding to the chill, so it's definitely crisp outside.
I'd suggest grabbing a warm layer or light coat before heading out, and you'll be
all set for a pleasant stroll.
```

```
$ python module_1/weather.py
In which city are you interested in the weather? Qwzxplk
Sorry, choose another city
```

## Обработка ошибок

| Ситуация | Что выводит скрипт |
|---|---|
| Пустой ввод | `Please enter a valid city name.` |
| Город не найден | `Sorry, choose another city` |
| Open-Meteo недоступен | `Error: weather service is unavailable (...)` |
| OpenRouter недоступен или превышен лимит бесплатной модели (429) | Сообщение об ошибке и сырые данные о погоде, чтобы ответ всё равно был полезен |
| Модель вернула пустой ответ | То же: сообщение об ошибке и сырые данные |

## Настройка

Константы в начале [`weather.py`](weather.py):

| Константа | По умолчанию | Что задаёт |
|---|---|---|
| `MODEL` | `nvidia/nemotron-3.5-lightning:free` | Любая модель с OpenRouter (суффикс `:free` у бесплатных) |
| `SYSTEM_PROMPT` | короткий дружелюбный ответ + совет | Стиль и формат ответа |
| `WEATHER_CODES` | коды WMO на английском | Текстовое описание погоды |

> Nemotron сначала рассуждает, а потом отвечает. Токены рассуждения тоже расходуют лимит, поэтому `max_tokens` стоит 2000: при 200 ответ приходил пустым.
