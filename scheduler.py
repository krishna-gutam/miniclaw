from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

scheduler = AsyncIOScheduler()
telegram_app = None   # filled by the bot on startup

def init_scheduler(app):
    """Start the scheduler and store the Telegram application instance."""
    global telegram_app
    telegram_app = app
    scheduler.start()

async def _send_reminder(chat_id: int, message: str):
    await telegram_app.bot.send_message(chat_id=chat_id, text=f"⏰ Reminder: {message}")

def add_reminder(chat_id: str, reminder_time_iso: str, message: str):
    """
    Schedule a one‑time reminder.
    :param chat_id: Telegram chat ID (string)
    :param reminder_time_iso: ISO‑8601 datetime string (e.g., "2026-07-10T15:00:00")
    :param message: Text to send
    """
    dt = datetime.fromisoformat(reminder_time_iso)
    job_id = f"reminder_{chat_id}_{dt.isoformat()}"
    scheduler.add_job(
        _send_reminder,
        'date',
        run_date=dt,
        args=[chat_id, message],
        id=job_id,
        replace_existing=True,
    )

def add_cron_job(chat_id: str, cron_expression: str, message: str):
    """
    Schedule a recurring reminder using a 5‑field cron expression.
    Example: "0 9 * * 1" = every Monday at 09:00.
    """
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        raise ValueError("Cron expression must have exactly 5 fields: minute hour day month day_of_week")
    job_id = f"cron_{chat_id}_{cron_expression}"
    scheduler.add_job(
        _send_reminder,
        'cron',
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
        args=[chat_id, message],
        id=job_id,
        replace_existing=True,
    )
