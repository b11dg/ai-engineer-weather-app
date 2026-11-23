# Модуль 4. Умный агент склада (Function Calling + GUI)

> Это задание модуля 4. Обзор всех модулей есть в [корневом README](../README.md).

Покупатель спрашивает в окне чата, есть ли товар в наличии. Модель сама остатков не знает, поэтому просит вызвать инструмент `check_inventory`. Программа перехватывает этот запрос (`tool_calls`), вызывает локальную mock-функцию, передаёт результат обратно сообщением с ролью `tool` и делает финальный запрос со `stream=True`. Ответ появляется в окне по мере генерации.

- **API:** [OpenRouter](https://openrouter.ai) через библиотеку `openai`
- **Модель:** NVIDIA Nemotron 3.5 Super VL, бесплатная версия (`nvidia/nemotron-3.5-super-vl:free`)
- **Схема инструмента:** генерируется из модели Pydantic
- **GUI:** PyQt6, работает на Windows, macOS и Linux
- **Файл:** [`smart_agent.py`](smart_agent.py)

## Почему эта модель

Среди бесплатных моделей OpenRouter нужна такая, которая поддерживает и `tools` (Function Calling), и `structured_outputs` (то есть `"strict": true`). Это проверено по `https://openrouter.ai/api/v1/models` → `supported_parameters`. Nemotron 3.5 Super VL поддерживает оба параметра и уже стабильно работала в модуле 3. Nemotron 3.5 Lightning и 3 Ultra из модулей 1–2 `tools` поддерживают, а `strict` нет.

В запросе передаётся `extra_body={"provider": {"require_parameters": True}}`: OpenRouter отправит его только тем провайдерам, которые поддерживают все наши параметры, а не тем, кто молча проигнорирует инструменты.

## Почему PyQt6

- Одна кодовая база для Windows, macOS и Linux, готовые сборки ставятся через `pip`, ничего не нужно собирать.
- Нативный вид окна на каждой ОС.
- `tkinter` не подошёл: в Python из Homebrew его нет без отдельного пакета `python-tk`.
- PySide6 (официальные биндинги Qt) тоже рассматривался, но под macOS 12 он собран только для Python до 3.13, а в проекте Python 3.15.

Версия закреплена на **6.9** (`PyQt6>=6.9,<6.10` в `requirements.txt`): Qt 6.10 и новее требует macOS 13+, а 6.9 ещё работает на macOS 12. Под Windows и Linux эта версия тоже есть.

## Как работает цикл Function Calling

```
Пользователь ──► [1] запрос с tools ──► модель
                                          │
                     tool_calls: check_inventory("macbook air"),
                                 check_inventory("airpods")
                                          │
      [2] программа перехватывает вызовы, проверяет аргументы через Pydantic
          и вызывает check_inventory() у себя локально
                                          │
      [3] результаты → сообщения {"role": "tool", "tool_call_id": ..., "content": ...}
                                          │
      [4] финальный запрос, stream=True ──► ответ по кусочкам в окно
```

1. **Первый запрос.** В `messages` лежат System Prompt и вопрос, в `tools` описание инструмента, `tool_choice="auto"`: модель сама решает, нужен ли инструмент.
2. **Проверка `tool_calls`.** Если модель вернула `tool_calls`, программа добавляет в историю сообщение ассистента с этими вызовами. Затем для каждого вызова проверяет аргументы и вызывает функцию. На вопрос о двух товарах модель делает два вызова сразу.
3. **Сообщения `tool`.** Результат каждого вызова добавляется с ролью `tool` и тем же `tool_call_id`, чтобы модель понимала, какой ответ к какому вызову относится.
4. **Финальный запрос со `stream=True`.** Модель формулирует ответ по данным со склада. `tools` остаются в запросе, потому что история на них ссылается, но `tool_choice="none"` просит только текст. Кусочки (`delta.content`) сразу выводятся в окно.

Если инструмент не нужен (например, «Привет, как дела?»), модель отвечает сразу на шаге 1, и программа просто показывает этот ответ.

## Mock-функция

```python
INVENTORY = {"iphone 15": 24, "macbook air": 5, "airpods": 0}

def check_inventory(item_name: str) -> str:
    item_name = item_name.strip()
    stock = INVENTORY.get(item_name.lower(), "Не знайдено")
    return f"Залишок товару {item_name}: {stock}"
```

## JSON-схема инструмента из Pydantic

Схема не пишется руками. Аргументы описываются моделью Pydantic, а `openai.pydantic_function_tool` превращает её в элемент массива `tools`:

```python
class CheckInventoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")   # → "additionalProperties": false
    item_name: str = Field(description="...")

TOOLS = [pydantic_function_tool(CheckInventoryArgs, name="check_inventory", description="...")]
```

Получается такая схема (её же видно в окне программы):

```json
{
  "type": "function",
  "function": {
    "name": "check_inventory",
    "strict": true,
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "item_name": {"description": "Name of exactly one product ...", "title": "Item Name", "type": "string"}
      },
      "required": ["item_name"],
      "title": "CheckInventoryArgs",
      "type": "object"
    },
    "description": "Checks how many units of one product are left ..."
  }
}
```

| Поле | Что делает |
|------|------------|
| `"strict": true` | Модель обязана сгенерировать аргументы строго по схеме (structured outputs) |
| `"additionalProperties": false` | Запрещает придумывать поля, которых нет в схеме. Получается из `extra="forbid"` |
| `"required"` | Все поля обязательны (`strict` этого требует), `pydantic_function_tool` проставляет это сам |
| `description` у инструмента и поля | Объясняет модели, когда вызывать инструмент (на каждый товар отдельно) и в каком виде передавать название: английское название латиницей, даже если пользователь написал «макбук эйр» |

У модели Pydantic намеренно нет docstring: Pydantic положил бы его в схему как `description`, и он ушёл бы в модель.

**Проверка ответа модели.** Аргументы, которые вернула модель, проверяются той же моделью Pydantic: `CheckInventoryArgs.model_validate_json(arguments)`. Если пришли лишние поля, нет `item_name` или JSON битый, программа не падает. Модель получает сообщение `tool` с текстом ошибки, например `Помилка: некоректні аргументи (extra: Extra inputs are not permitted).` Неизвестное имя инструмента тоже обрабатывается.

### Почему без `response_format: json_schema`

`response_format` с `json_schema` задаёт формат **ответа** модели. Он нужен, когда данные генерирует сама модель (как тикет в модуле 3). Здесь данные приходят из нашей функции `check_inventory`, а ответ пользователю — обычный текст, поэтому строгая схема нужна только для **аргументов инструмента**. Её и дают `"strict": true` и `"additionalProperties": false` в `tools`.

## Окно программы

- **Слева чат:** поле ввода с уже вписанным вопросом из задания, кнопки «Отправить» (или Enter) и «Новый диалог». История сохраняется между сообщениями, поэтому можно задавать уточняющие вопросы.
- **Справа журнал агента:** каждый шаг цикла (запрос, `tool_calls` с аргументами, результат функции, финальный стриминг). Под журналом схема `tools` и таблица mock-склада.
- Запрос к API идёт в отдельном потоке (`QThread`), поэтому окно не подвисает, а текст ответа появляется по мере стриминга.
- Если `MODEL_API_KEY` нет, программа покажет окно с подсказкой про `.env`.
- Ошибки API (нет сети, лимит 429 у бесплатной модели) и пустой ответ показываются в чате красным. Если запрос не удался, история диалога не портится.

## Запуск

Команды выполняются из корня проекта, окружение общее.

```bash
# 1. Виртуальное окружение и зависимости (если ещё не сделано)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Ключ OpenRouter в корневом .env
cp .env.example .env
# впишите ключ в MODEL_API_KEY=...

# 3. Окно программы
python module_4/smart_agent.py

# или без окна, в терминале (по умолчанию вопрос из задания)
python module_4/smart_agent.py --cli
python module_4/smart_agent.py --cli "Есть айфон 15 и айрподсы?"
```

> На Linux для Qt могут понадобиться системные библиотеки XCB. Если окно не открывается с ошибкой `xcb`, установите `libxcb-cursor0` (Debian/Ubuntu: `sudo apt install libxcb-cursor0`).

## Пример (реальный запуск, `--cli`)

```
Вопрос: Привіт! Чи є у вас в наявності macbook air та airpods?

→ Запрос к модели nvidia/nemotron-3.5-super-vl:free с инструментом check_inventory (tool_choice=auto)
← Модель вернула tool_calls: 2
   ⚙ check_inventory({"item_name":"macbook air"})
   ✓ Залишок товару macbook air: 5
   ⚙ check_inventory({"item_name":"airpods"})
   ✓ Залишок товару airpods: 0
→ Финальный запрос с результатами (stream=True)

Ответ: MacBook Air є в наявності — 5 одиниць. AirPods зараз відсутні.

← Ответ получен потоком: 9 фрагментов
```

Другие проверки:

| Вопрос | Что сделала модель | Ответ |
|--------|--------------------|-------|
| «Привет, как дела?» | Инструмент не вызывала | Ответила сразу |
| «А iPhone 16 у вас есть?» | `check_inventory("iPhone 16")` → `Не знайдено` | «К сожалению, iPhone 16 мы не продаём.» |
| «Есть айфон 15 и айрподсы?» | Два вызова: `iphone 15` → 24, `airpods` → 0 | «Айфон 15 есть — 24 штуки. Айрподсы сейчас нет в наличии.» |
