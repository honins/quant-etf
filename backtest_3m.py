import os

from backtest_recent import main as backtest_recent_main


def main():
    os.environ.setdefault("LOOKBACK_DAYS", "90")
    backtest_recent_main()


if __name__ == "__main__":
    main()
