from src.core.daily_report_service import generate_daily_report


def main() -> None:
    try:
        result = generate_daily_report(send_notification=True)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if result["notification_status"] == "failed":
        print(f"Notification failed: {result['notification_error']}")
    elif result["notification_status"] == "skipped":
        print("Notification skipped: FEISHU_WEBHOOK is not configured.")

    print(f"Daily report ready: {result['report_path']}")
    print("All tasks completed.")


if __name__ == "__main__":
    main()
