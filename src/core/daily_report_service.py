from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.dashboard.data_builder import build_dashboard_payload
from src.utils.feishu_bot import FeishuBot
from src.utils.reporter import Reporter


def _send_report_notification(report_path: Path) -> str:
    bot = FeishuBot()
    if not bot.webhook:
        return "skipped"

    with report_path.open("r", encoding="utf-8") as handle:
        content = handle.read()

    title = f"Quant-ETF Daily Report ({datetime.now().strftime('%Y-%m-%d')})"
    bot.send_markdown(title, content)
    return "sent"


def generate_daily_report(send_notification: bool = False, history_days: int = 120) -> dict:
    dashboard_payload = build_dashboard_payload(history_days=history_days)
    reporter = Reporter()
    report_path = Path(
        reporter.generate_markdown(
            [dict(item) for item in dashboard_payload["signals"]["all"]],
            dashboard_payload["market_status"],
            dashboard_payload["holdings"],
        )
    )

    notification_status = "not_requested"
    notification_error = None
    if send_notification:
        try:
            notification_status = _send_report_notification(report_path)
        except Exception as exc:
            notification_status = "failed"
            notification_error = str(exc)

    return {
        "report_path": str(report_path),
        "report_url": report_path.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_status": dashboard_payload["market_status"],
        "model_name": dashboard_payload["model_name"],
        "buy_count": len(dashboard_payload["signals"]["buy"]),
        "holdings_count": len(dashboard_payload["holdings"]),
        "notification_status": notification_status,
        "notification_error": notification_error,
    }
