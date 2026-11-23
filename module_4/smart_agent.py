"""
Smart inventory agent (module 4): Function Calling with a GUI.

The user asks about products in a chat window. The model does not know the
stock itself, so it asks to call the check_inventory tool (tool_calls).
The program intercepts that request, runs the local mock function, sends
the result back in a message with role "tool" and makes the final request
with stream=True, so the answer appears in the window as it is generated.

The tool's JSON Schema is not written by hand: it is generated from the
Pydantic model CheckInventoryArgs by openai.pydantic_function_tool, which
sets "strict": true and "additionalProperties": false. The same Pydantic
model validates the arguments the model sends back.

Run:
    python module_4/smart_agent.py                 # GUI window
    python module_4/smart_agent.py --cli "питання"  # terminal, no window

API:    OpenRouter via the openai library (key in ../.env as MODEL_API_KEY)
Model:  NVIDIA Nemotron 3.5 Super VL, free tier (supports tools + strict)
GUI:    PyQt6 (Windows, macOS, Linux)
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type

from dotenv import load_dotenv
from openai import APIError, OpenAI, pydantic_function_tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3.5-super-vl:free"
MAX_TOKENS = 1000

# Send the request only to providers that support every parameter we use
# (tools, tool_choice, strict), otherwise OpenRouter may pick one that
# silently ignores the tools.
EXTRA_BODY = {"provider": {"require_parameters": True}}

DEFAULT_QUESTION = "Привіт! Чи є у вас в наявності macbook air та airpods?"

SYSTEM_PROMPT = """\
You are a friendly sales assistant of a small electronics store.

- You do not know the current stock. Whenever the user asks whether a
  product is available or how many units are left, call the
  check_inventory tool. Call it once per product; if the user asks about
  several products, make several calls in the same turn.
- Answer only from the tool results, never guess. A number is the units in
  stock; 0 means the product is out of stock right now; "Не знайдено" means
  the store does not sell this product.
- Do not promise prices, discounts or delivery dates.
- Reply in the language of the user's latest message: Russian to Russian,
  Ukrainian to Ukrainian, English to English. Tool results are always in
  Ukrainian; that does not change the reply language.
- Answer briefly, in plain text without Markdown.
"""


# --- The tool: a mock function and its schema --------------------------------

INVENTORY = {"iphone 15": 24, "macbook air": 5, "airpods": 0}


def check_inventory(item_name: str) -> str:
    """Mock warehouse lookup: returns the stock of one product as text."""
    item_name = item_name.strip()
    stock = INVENTORY.get(item_name.lower(), "Не знайдено")
    return f"Залишок товару {item_name}: {stock}"


# Arguments of check_inventory; the tool's JSON Schema is built from this model.
# (No docstring on purpose: Pydantic would put it into the schema the model sees.)
class CheckInventoryArgs(BaseModel):
    # extra="forbid" becomes "additionalProperties": false in the schema,
    # and model_validate_json rejects unknown fields.
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(
        description=(
            "Name of exactly one product as it is spelled in the store "
            "catalog: the official English name in Latin letters, without "
            "quantities or extra words, for example 'macbook air', "
            "'iphone 15' or 'airpods'. If the user writes it in Cyrillic or "
            "slang ('макбук эйр', 'айрподсы'), convert it to the English "
            "name. Case does not matter."
        ),
    )


# pydantic_function_tool turns the Pydantic model into a tools-array entry
# with "strict": true, "additionalProperties": false and every field required.
TOOLS = [
    pydantic_function_tool(
        CheckInventoryArgs,
        name="check_inventory",
        description=(
            "Checks how many units of one product are left in the store's "
            "warehouse right now. Use it for any question about availability "
            "or stock, one call per product. Returns a line like "
            "'Залишок товару <name>: <units>'; 0 means out of stock, "
            "'Не знайдено' means the store does not sell this product."
        ),
    )
]

# Tool name -> (Python function, Pydantic model of its arguments).
AVAILABLE_FUNCTIONS: Dict[str, Tuple[Callable[..., str], Type[BaseModel]]] = {
    "check_inventory": (check_inventory, CheckInventoryArgs),
}


# --- The agent loop (no GUI here, so it also runs in the terminal) -----------

class AgentError(Exception):
    """The model answered, but not with something we can use."""


# on_event(kind, text): "step" is a line for the agent log,
# "chunk" is a piece of the answer for the user.
EventHandler = Callable[[str, str], None]


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=MODEL_API_KEY)


def new_conversation() -> List[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def execute_tool_call(name: str, arguments: str) -> str:
    """Runs one tool call. Errors are returned as text for the model, not raised."""
    if name not in AVAILABLE_FUNCTIONS:
        return f"Помилка: невідомий інструмент '{name}'."
    function, args_model = AVAILABLE_FUNCTIONS[name]
    try:
        args = args_model.model_validate_json(arguments or "")
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
            for err in e.errors(include_url=False)
        )
        return f"Помилка: некоректні аргументи ({problems})."
    return function(**args.model_dump())


def run_agent(client: OpenAI, messages: List[dict], on_event: EventHandler) -> str:
    """
    One user turn. `messages` must already end with the user's message;
    the assistant and tool messages are appended to it. Returns the answer.
    """
    # 1. First request: the model sees the tools and decides whether to call them.
    on_event("step", f"→ Запрос к модели {MODEL} с инструментом check_inventory (tool_choice=auto)")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
        max_tokens=MAX_TOKENS,
        extra_body=EXTRA_BODY,
    )
    if not response.choices:
        raise AgentError("модель не вернула ответа")
    message = response.choices[0].message

    # 2. No tool_calls: the model answered right away (e.g. "Привет, как дела?").
    if not message.tool_calls:
        answer = (message.content or "").strip()
        if not answer:
            raise AgentError("модель вернула пустой ответ")
        on_event("step", "← Модель ответила сразу, инструмент не понадобился")
        on_event("chunk", answer)
        messages.append({"role": "assistant", "content": answer})
        return answer

    # 3. tool_calls: intercept them, run our function, add role="tool" messages.
    on_event("step", f"← Модель вернула tool_calls: {len(message.tool_calls)}")
    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    })
    for call in message.tool_calls:
        on_event("step", f"   ⚙ {call.function.name}({call.function.arguments})")
        result = execute_tool_call(call.function.name, call.function.arguments)
        on_event("step", f"   ✓ {result}")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # 4. Final request with the tool results, streamed piece by piece.
    # The tools stay in the request (the history mentions them), but
    # tool_choice="none" asks for a text answer only.
    on_event("step", "→ Финальный запрос с результатами (stream=True)")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
        temperature=0,
        max_tokens=MAX_TOKENS,
        stream=True,
        extra_body=EXTRA_BODY,
    )
    parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            on_event("chunk", delta)
    answer = "".join(parts).strip()
    if not answer:
        raise AgentError("модель вернула пустой финальный ответ")
    on_event("step", f"← Ответ получен потоком: {len(parts)} фрагментов")
    messages.append({"role": "assistant", "content": answer})
    return answer


def describe_error(error: Exception) -> str:
    if isinstance(error, APIError):
        status = getattr(error, "status_code", None)
        if status == 429:
            return "лимит запросов бесплатной модели (429). Подождите минуту и попробуйте снова."
        return f"ошибка API: {error}"
    if isinstance(error, AgentError):
        return str(error)
    return f"неожиданная ошибка: {error!r}"


# --- Terminal mode -----------------------------------------------------------

def run_cli(question: str) -> int:
    messages = new_conversation() + [{"role": "user", "content": question}]
    print(f"Вопрос: {question}\n")

    in_answer = False  # are we in the middle of printing streamed chunks?

    def on_event(kind: str, text: str) -> None:
        nonlocal in_answer
        if kind == "chunk":
            if not in_answer:
                print("\nОтвет: ", end="")
                in_answer = True
            print(text, end="", flush=True)
        else:
            if in_answer:
                print("\n")
                in_answer = False
            print(text, flush=True)

    try:
        run_agent(make_client(), messages, on_event)
    except (APIError, AgentError) as e:
        print(f"\nОшибка: {describe_error(e)}")
        return 1
    if in_answer:
        print()
    return 0


# --- GUI mode (PyQt6) --------------------------------------------------------

def run_gui() -> int:
    # Imported here, so --cli works even without a display.
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFontDatabase, QTextCharFormat, QTextCursor
    from PyQt6.QtWidgets import (
        QApplication, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
        QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
        QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
    )

    class AgentWorker(QThread):
        """Runs one agent turn off the UI thread; talks to the window via signals."""

        step = pyqtSignal(str)
        chunk = pyqtSignal(str)
        succeeded = pyqtSignal(object)  # the updated message list
        failed = pyqtSignal(str)

        def __init__(self, client: OpenAI, messages: List[dict]):
            super().__init__()
            self.client = client
            self.messages = messages

        def run(self) -> None:
            def on_event(kind: str, text: str) -> None:
                (self.chunk if kind == "chunk" else self.step).emit(text)

            try:
                run_agent(self.client, self.messages, on_event)
            except Exception as e:  # shown in the window instead of crashing
                self.failed.emit(describe_error(e))
            else:
                self.succeeded.emit(self.messages)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Smart Agent: склад магазина")
            self.resize(1100, 650)
            self.client = make_client() if MODEL_API_KEY else None
            self.history = new_conversation()
            self.worker = None

            mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

            # Left: chat.
            self.chat_view = QTextBrowser()
            self.chat_view.setOpenExternalLinks(True)
            self.input = QLineEdit(DEFAULT_QUESTION)
            self.input.setPlaceholderText("Спросите о наличии товара…")
            self.input.returnPressed.connect(self.send)
            self.send_button = QPushButton("Отправить")
            self.send_button.setDefault(True)
            self.send_button.clicked.connect(self.send)
            self.reset_button = QPushButton("Новый диалог")
            self.reset_button.clicked.connect(self.reset)

            input_row = QHBoxLayout()
            input_row.addWidget(self.input, 1)
            input_row.addWidget(self.send_button)
            input_row.addWidget(self.reset_button)

            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.addWidget(QLabel("<b>Чат</b>"))
            left_layout.addWidget(self.chat_view, 1)
            left_layout.addLayout(input_row)

            # Right: agent log, tool schema and the mock warehouse.
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setFont(mono)

            schema_view = QPlainTextEdit(json.dumps(TOOLS, indent=2, ensure_ascii=False))
            schema_view.setReadOnly(True)
            schema_view.setFont(mono)

            table = QTableWidget(len(INVENTORY), 2)
            table.setHorizontalHeaderLabels(["Товар", "Остаток"])
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for row, (name, stock) in enumerate(INVENTORY.items()):
                table.setItem(row, 0, QTableWidgetItem(name))
                table.setItem(row, 1, QTableWidgetItem(str(stock)))
            table.setFixedHeight(
                table.verticalHeader().defaultSectionSize() * len(INVENTORY)
                + table.horizontalHeader().sizeHint().height()
                + 2 * table.frameWidth()
            )

            right_split = QSplitter(Qt.Orientation.Vertical)
            for title, widget in (
                ("Журнал агента", self.log_view),
                ("Схема инструмента (tools, сгенерирована из Pydantic)", schema_view),
                ("Склад (mock-данные check_inventory)", table),
            ):
                box = QWidget()
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(0, 0, 0, 0)
                box_layout.addWidget(QLabel(f"<b>{title}</b>"))
                box_layout.addWidget(widget)
                right_split.addWidget(box)
            right_split.setSizes([330, 220, 120])

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(left)
            splitter.addWidget(right_split)
            splitter.setSizes([600, 500])
            self.setCentralWidget(splitter)
            self.statusBar().showMessage(f"Модель: {MODEL}")

            if self.client is None:
                self.send_button.setEnabled(False)
                self.input.setEnabled(False)

        # Chat helpers.
        def append_message(self, who: str, text: str, color: str = "") -> None:
            style = f' style="color:{color}"' if color else ""
            body = html.escape(text).replace("\n", "<br>")
            self.chat_view.append(f"<p{style}><b>{who}:</b> {body}</p>")

        def append_chunk(self, text: str) -> None:
            cursor = self.chat_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(text, QTextCharFormat())
            self.chat_view.setTextCursor(cursor)
            self.chat_view.ensureCursorVisible()

        def log(self, text: str) -> None:
            self.log_view.appendPlainText(text)

        def set_busy(self, busy: bool) -> None:
            self.send_button.setEnabled(not busy)
            self.reset_button.setEnabled(not busy)
            self.input.setEnabled(not busy)
            self.statusBar().showMessage("Модель думает…" if busy else f"Модель: {MODEL}")

        # Actions.
        def send(self) -> None:
            question = self.input.text().strip()
            if not question or self.worker is not None:
                return
            self.input.clear()
            self.append_message("Вы", question)
            self.chat_view.append("<b>Ассистент:</b> ")
            if self.log_view.toPlainText():
                self.log("")
            self.log(f"=== Вопрос: {question}")
            self.set_busy(True)

            # The worker gets a copy: if the request fails, the history stays clean.
            messages = self.history + [{"role": "user", "content": question}]
            self.worker = AgentWorker(self.client, messages)
            self.worker.step.connect(self.log)
            self.worker.chunk.connect(self.append_chunk)
            self.worker.succeeded.connect(self.on_success)
            self.worker.failed.connect(self.on_failure)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()

        def on_success(self, messages: List[dict]) -> None:
            self.history = messages

        def on_failure(self, error: str) -> None:
            self.append_message("Ошибка", error, color="#c0392b")
            self.log(f"✗ Ошибка: {error}")

        def on_worker_finished(self) -> None:
            self.worker.deleteLater()
            self.worker = None
            self.set_busy(False)
            self.input.setFocus()

        def reset(self) -> None:
            self.history = new_conversation()
            self.chat_view.clear()
            self.log_view.clear()
            self.input.setText(DEFAULT_QUESTION)

        def closeEvent(self, event) -> None:
            if self.worker is not None:
                self.worker.wait(3000)
            super().closeEvent(event)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if not MODEL_API_KEY:
        QMessageBox.critical(
            window,
            "Нет ключа API",
            "MODEL_API_KEY не найден.\nСкопируйте .env.example в .env в корне "
            "репозитория и впишите туда ключ OpenRouter.",
        )
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart inventory agent (module 4)")
    parser.add_argument(
        "--cli", metavar="QUESTION", nargs="?", const=DEFAULT_QUESTION,
        help="run in the terminal without a window (default question if omitted)",
    )
    args = parser.parse_args()

    if args.cli is not None:
        if not MODEL_API_KEY:
            print("Ошибка: MODEL_API_KEY не найден в .env")
            return 1
        return run_cli(args.cli)
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
