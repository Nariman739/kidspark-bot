#!/usr/bin/env python3
"""
Kids Park AI Consultant — Multi-Agent Telegram Bot
@KidsPark_KRG_bot

Architecture:
  1. ROUTER (haiku, fast) — classifies message → picks category
  2. SPECIALIST (sonnet, smart) — answers using only relevant KB section
  3. ESCALATION — notifies manager for bookings/complaints
"""

import asyncio
import logging
import os
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from knowledge_base import CATEGORY_KB, CATEGORIES_DESCRIPTION

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────
BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "anthropic/claude-haiku-4.5")
SPECIALIST_MODEL = os.environ.get("SPECIALIST_MODEL", "anthropic/claude-sonnet-4")
MANAGER_CHAT_ID = os.environ.get("MANAGER_CHAT_ID", "")

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Conversation Memory ──────────────────────────────────────────────
conversations: dict[int, list[dict]] = defaultdict(list)
MAX_HISTORY = 10

# ── Message Batching (debounce) ─────────────────────────────────────
# Collect multiple rapid messages into one before processing
DEBOUNCE_SECONDS = 3.0  # Wait 3s after last message
message_buffers: dict[int, list[str]] = defaultdict(list)
debounce_tasks: dict[int, asyncio.Task] = {}
debounce_contexts: dict[int, tuple] = {}  # (update, context) for delayed processing


def get_history(chat_id: int) -> list[dict]:
    return conversations[chat_id][-MAX_HISTORY:]


def add_message(chat_id: int, role: str, content: str):
    conversations[chat_id].append({"role": role, "content": content})
    if len(conversations[chat_id]) > MAX_HISTORY * 2:
        conversations[chat_id] = conversations[chat_id][-MAX_HISTORY:]


# ── AGENT 1: Router (fast, cheap) ────────────────────────────────────
ROUTER_PROMPT = f"""Ты классификатор сообщений для детского развлекательного центра Kids Park.

Определи категорию сообщения клиента. Верни ТОЛЬКО одно слово — название категории.

Категории:
{CATEGORIES_DESCRIPTION}

ВАЖНО: Учитывай контекст предыдущих сообщений. Если клиент уже обсуждал тему и продолжает — используй ту же категорию.

Верни ОДНО слово: general, entrance, attractions, birthday, booking, menu, drinks, ramadan, vacancy, complaint, other"""


async def route_message(chat_id: int, user_message: str) -> str:
    """Classify message into a category using fast/cheap model."""
    history = get_history(chat_id)

    # Build context summary from recent messages
    context_lines = []
    for msg in history[-4:]:
        role = "Клиент" if msg["role"] == "user" else "Бот"
        context_lines.append(f"{role}: {msg['content'][:100]}")
    context = "\n".join(context_lines) if context_lines else "Новый диалог"

    try:
        response = ai_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Контекст диалога:\n{context}\n\nНовое сообщение клиента: {user_message}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        category = response.choices[0].message.content.strip().lower()
        # Clean up — extract first word only
        category = category.split()[0].strip(".,!\"'")
        if category not in CATEGORY_KB:
            category = "other"
        logger.info(f"Router: '{user_message[:50]}' → {category}")
        return category
    except Exception as e:
        logger.error(f"Router error: {e}")
        return "other"


# ── AGENT 2: Specialist (smart, focused) ─────────────────────────────
SPECIALIST_PROMPT_TEMPLATE = """Ты — дружелюбный консультант детского центра Kids Park в Караганде.

СЕГОДНЯ: {today_date} ({today_weekday})

СТИЛЬ ОБЩЕНИЯ:
- Отвечай на языке клиента (русский → русский, казахский → казахский)
- Будь кратким! 2-4 предложения максимум, не пиши простыни
- Если вопрос общий ("сколько стоит вход?") — сначала уточни: возраст ребёнка, будний день или выходной
- Не вываливай ВСЮ информацию. Дай ответ на конкретный вопрос
- Если клиент не уточнил деталей — задай 1-2 уточняющих вопроса
- Используй эмодзи умеренно (1-2 на сообщение)
- Говори как живой человек, не как робот
- Когда клиент говорит "завтра", "послезавтра", "в субботу" — точно определяй дату и день недели по текущей дате

ПРАВИЛА:
- Никогда не выдумывай информацию. Только то, что есть в базе знаний
- Если клиент хочет ЗАБРОНИРОВАТЬ — скажи "Сейчас передам менеджеру, он свяжется с вами!" и добавь в конец ответа тег [MANAGER]
- Если клиент просит связать с менеджером или живым человеком — добавь [MANAGER]
- Если клиент жалуется серьёзно — добавь [MANAGER]
- Если не знаешь ответа — добавь [MANAGER]
- НЕ добавляй [MANAGER] просто при вопросах о ценах, меню, графике и т.д.

БАЗА ЗНАНИЙ:
{kb_section}"""

WEEKDAYS_RU = {
    0: "понедельник", 1: "вторник", 2: "среда", 3: "четверг",
    4: "пятница", 5: "суббота", 6: "воскресенье",
}


async def specialist_respond(chat_id: int, user_message: str, category: str) -> dict:
    """Generate response using specialist with relevant KB section."""
    kb_section = CATEGORY_KB.get(category, CATEGORY_KB["other"])
    now = datetime.now()
    system_prompt = SPECIALIST_PROMPT_TEMPLATE.format(
        kb_section=kb_section,
        today_date=now.strftime("%d.%m.%Y"),
        today_weekday=WEEKDAYS_RU[now.weekday()],
    )

    history = get_history(chat_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = ai_client.chat.completions.create(
            model=SPECIALIST_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,  # Короткие ответы!
        )
        raw = response.choices[0].message.content.strip()

        # Check for manager escalation tag
        needs_manager = "[MANAGER]" in raw
        clean_response = raw.replace("[MANAGER]", "").strip()

        return {
            "response": clean_response,
            "needs_manager": needs_manager,
            "category": category,
        }
    except Exception as e:
        logger.error(f"Specialist error: {e}")
        return {
            "response": "Извините, произошла небольшая ошибка. Попробуйте ещё раз через минуту! 🙏",
            "needs_manager": False,
            "category": category,
        }


# ── Notify Manager ───────────────────────────────────────────────────
async def notify_manager(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user, category: str):
    if not MANAGER_CHAT_ID:
        logger.warning("MANAGER_CHAT_ID not set")
        return

    username = f"@{user.username}" if user.username else f"{user.first_name} {user.last_name or ''}".strip()
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Last few messages for context
    history = get_history(chat_id)
    recent = "\n".join(
        f"{'👤' if m['role'] == 'user' else '🤖'} {m['content'][:150]}"
        for m in history[-6:]
    )

    text = (
        f"🔔 Нужен менеджер!\n\n"
        f"👤 {username} (ID: {chat_id})\n"
        f"🕐 {now}\n"
        f"📂 Тема: {category}\n\n"
        f"Последние сообщения:\n{recent}"
    )

    try:
        await context.bot.send_message(int(MANAGER_CHAT_ID), text)
    except Exception as e:
        logger.error(f"Failed to notify manager: {e}")


# ── Bot Handlers ─────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversations[chat_id] = []
    logger.info(f"/start from {chat_id}")

    await update.message.reply_text(
        "Здравствуйте! 👋\n\n"
        "Я — консультант Kids Park 🎉\n"
        "Задайте любой вопрос — помогу с ценами, пакетами, меню и всем остальным!\n\n"
        "Сәлеметсіз бе! Kids Park кеңесшісімін 🎈\n"
        "Сұрақтарыңызға жауап беремін!"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто напишите вопрос! Например:\n"
        "• Сколько стоит вход?\n"
        "• Расскажи про ДР пакеты\n"
        "• Что есть в меню?\n"
        "• Где вы находитесь?\n\n"
        "/start — начать заново\n"
        "/manager — связаться с менеджером"
    )


async def cmd_manager(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "Менеджер свяжется с вами в ближайшее время! 🙏\n"
        "Срочно: 8 778 268 27 79"
    )
    await notify_manager(context, chat_id, update.effective_user, "manual_request")


async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Ваш Chat ID: `{chat_id}`", parse_mode="Markdown")


async def process_batched_messages(chat_id: int, user, bot_context: ContextTypes.DEFAULT_TYPE):
    """Process all collected messages after debounce timer expires."""
    await asyncio.sleep(DEBOUNCE_SECONDS)

    # Grab all buffered messages and clear
    messages = message_buffers.pop(chat_id, [])
    debounce_tasks.pop(chat_id, None)
    debounce_contexts.pop(chat_id, None)

    if not messages:
        return

    # Combine multiple messages into one
    combined_text = " ".join(messages)
    logger.info(f"[{chat_id}] Batched {len(messages)} msg(s): '{combined_text[:80]}'")

    await bot_context.bot.send_chat_action(chat_id, "typing")

    # Step 1: ROUTER — classify message (fast, cheap)
    category = await route_message(chat_id, combined_text)

    # Step 2: SPECIALIST — generate response (smart, focused KB)
    result = await specialist_respond(chat_id, combined_text, category)

    # Step 3: Save history
    add_message(chat_id, "user", combined_text)
    add_message(chat_id, "assistant", result["response"])

    # Step 4: Send response
    await bot_context.bot.send_message(chat_id, result["response"])

    # Step 5: Escalate if needed
    if result["needs_manager"]:
        await notify_manager(bot_context, chat_id, user, category)
        logger.info(f"[{chat_id}] Escalated to manager ({category})")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user = update.effective_user
    user_text = update.message.text.strip()
    if not user_text:
        return

    logger.info(f"[{chat_id}] Received: {user_text[:80]}")

    # Add message to buffer
    message_buffers[chat_id].append(user_text)

    # Cancel previous debounce timer if exists
    if chat_id in debounce_tasks:
        debounce_tasks[chat_id].cancel()

    # Start new debounce timer
    debounce_tasks[chat_id] = asyncio.create_task(
        process_batched_messages(chat_id, user, context)
    )


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is required!")
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_KEY_HERE":
        logger.warning("OPENROUTER_API_KEY not set!")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("manager", cmd_manager))
    app.add_handler(CommandHandler("register", cmd_register))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"🚀 Kids Park Bot starting (multi-agent)")
    logger.info(f"Router: {ROUTER_MODEL} | Specialist: {SPECIALIST_MODEL}")
    logger.info(f"Manager: {MANAGER_CHAT_ID or 'NOT SET'}")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
