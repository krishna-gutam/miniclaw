import os
from dotenv import load_dotenv

from fastapi import FastAPI

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
)

from langchain_core.messages import AIMessage

from agent import (
    graph,
    sanitize_content,
    get_pending_tool_calls,
    format_tool_calls_for_approval,
)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is not set.")

app = FastAPI()

APPROVAL_WORDS = {
    "yes",
    "y",
    "approve",
    "approved",
    "ok",
    "okay",
    "go ahead",
    "continue",
    "proceed",
}


def is_approval_message(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in APPROVAL_WORDS


async def send_long_message(update: Update, text: str, chunk_size: int = 4000):
    """
    Telegram has message length limits.
    Send long responses in chunks.
    """
    if not text:
        text = "Done."

    for i in range(0, len(text), chunk_size):
        await update.message.reply_text(text[i : i + chunk_size])


def get_thread_config(chat_id: str):
    return {
        "configurable": {
            "thread_id": chat_id,
        }
    }


def get_saved_messages(config) -> list:
    """
    Read saved LangGraph state for this Telegram chat.
    """
    snapshot = graph.get_state(config)

    if not snapshot or not snapshot.values:
        return []

    return snapshot.values.get("messages", [])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    chat_id = str(update.message.chat_id)
    config = get_thread_config(chat_id)

    saved_messages = get_saved_messages(config)
    pending_tool_calls = get_pending_tool_calls(saved_messages)

    if is_approval_message(user_text) and pending_tool_calls:
        result = graph.invoke(
            {
                "approved": True,
                "chat_id": chat_id,
            },
            config=config,
        )
    else:
        result = graph.invoke(
            {
                "messages": [("user", user_text)],
                "approved": False,
                "chat_id": chat_id,
            },
            config=config,
        )

    messages = result.get("messages", [])

    if not messages:
        await update.message.reply_text("No response generated.")
        return

    # Find the last AIMessage to represent the final assistant reply
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    # If the assistant is requesting tool execution, ask for approval
    if last_ai_message and last_ai_message.tool_calls:
        approval_text = format_tool_calls_for_approval(last_ai_message.tool_calls)
        await send_long_message(update, approval_text)
        return

    # Otherwise, send the assistant's text response
    if last_ai_message:
        response_text = sanitize_content(last_ai_message.content)
    else:
        # Fallback: send the content of the very last message
        response_text = sanitize_content(messages[-1].content)

    await send_long_message(update, response_text)


from scheduler import init_scheduler, scheduler

def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register a post_init callback that starts the scheduler
    async def post_init(app):
        init_scheduler(app)

    application.post_init = post_init

    async def post_shutdown(app):
        scheduler.shutdown(wait=False)

    application.post_shutdown = post_shutdown

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print("Bot is running...")
    application.run_polling()


if __name__ == "__main__":
    run_bot()