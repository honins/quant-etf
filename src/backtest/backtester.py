import numpy as np
import pandas as pd

from config.settings import settings
from src.backtest.strategy_config import StrategyConfig
from src.strategy.logic import RiskManager


class Backtester:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager()

    def run(
        self,
        df: pd.DataFrame,
        probs: np.ndarray,
        threshold=0.6,
        code: str = "",
        exit_probs: np.ndarray | None = None,
        config: StrategyConfig | None = None,
    ) -> dict:
        cash = self.initial_capital
        position = 0
        equity_curve = []
        daily_pnl = []
        position_history = []
        trades = []
        trailing_stop = 0.0
        entry_price = 0.0
        entry_date = None
        cumulative_turnover = 0.0
        previous_equity = self.initial_capital

        config = config or StrategyConfig.from_settings()
        exit_probs = probs if exit_probs is None else exit_probs

        peak_equity = self.initial_capital
        multiplier = config.atr_multiplier_aggressive if code in settings.AGGRESSIVE_TICKERS else config.atr_multiplier

        for i in range(len(df) - 1):
            date = df.iloc[i]["trade_date"]
            close_price = df.iloc[i]["close"]
            atr = df.iloc[i]["atr"]
            entry_score = probs[i]
            exit_score = exit_probs[i]

            next_open = df.iloc[i + 1]["open"]
            next_low = df.iloc[i + 1]["low"]
            next_date = df.iloc[i + 1]["trade_date"]

            current_equity = cash + position * close_price
            equity_curve.append({"date": date, "equity": current_equity})
            daily_pnl.append({"date": date, "pnl": current_equity - previous_equity})
            position_history.append(
                {
                    "date": date,
                    "position": int(position),
                    "cash": float(cash),
                    "close": float(close_price),
                    "exposure": float(position * close_price),
                    "weight": float((position * close_price) / current_equity) if current_equity > 0 else 0.0,
                }
            )
            previous_equity = current_equity
            if current_equity > peak_equity:
                peak_equity = current_equity

            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
            if drawdown >= config.max_drawdown_stop:
                if position > 0:
                    sell_price = next_open
                    turnover_value = position * sell_price
                    cash += position * sell_price
                    cumulative_turnover += turnover_value / self.initial_capital
                    trades.append(
                        {
                            "date": next_date,
                            "action": "SELL (MaxDD)",
                            "price": sell_price,
                            "pnl": (sell_price - entry_price) * position,
                            "shares": int(position),
                            "holding_days": self._holding_days(entry_date, next_date),
                        }
                    )
                    position = 0
                    trailing_stop = 0.0
                    entry_date = None
                break

            if position > 0:
                if next_low < trailing_stop:
                    sell_price = min(next_open, trailing_stop)
                    turnover_value = position * sell_price
                    cash += position * sell_price
                    cumulative_turnover += turnover_value / self.initial_capital
                    trades.append(
                        {
                            "date": next_date,
                            "action": "SELL (Stop)",
                            "price": sell_price,
                            "pnl": (sell_price - entry_price) * position,
                            "shares": int(position),
                            "holding_days": self._holding_days(entry_date, next_date),
                        }
                    )
                    position = 0
                    trailing_stop = 0.0
                    entry_date = None
                    continue

                if exit_score is not None and np.isfinite(exit_score) and exit_score < config.signal_exit_threshold:
                    sell_price = next_open
                    turnover_value = position * sell_price
                    cash += position * sell_price
                    cumulative_turnover += turnover_value / self.initial_capital
                    trades.append(
                        {
                            "date": next_date,
                            "action": "SELL (Signal)",
                            "price": sell_price,
                            "pnl": (sell_price - entry_price) * position,
                            "shares": int(position),
                            "holding_days": self._holding_days(entry_date, next_date),
                        }
                    )
                    position = 0
                    trailing_stop = 0.0
                    entry_date = None
                    continue

                window_start = max(0, i - config.exit_lookback_period + 1)
                recent_high = df["high"].iloc[window_start : i + 1].max()
                new_stop = recent_high - (multiplier * atr)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop

            if position == 0:
                if entry_score is not None and np.isfinite(entry_score) and entry_score > 0 and entry_score >= threshold:
                    buy_price = next_open
                    shares = int((cash * 0.99) / buy_price / 100) * 100
                    if shares > 0:
                        position = shares
                        cash -= shares * buy_price
                        cumulative_turnover += (shares * buy_price) / self.initial_capital
                        entry_price = buy_price
                        entry_date = next_date
                        trailing_stop = buy_price - (multiplier * atr)
                        trades.append(
                            {
                                "date": next_date,
                                "action": "BUY",
                                "price": buy_price,
                                "score": entry_score,
                                "shares": int(shares),
                            }
                        )

        final_equity = cash + position * df.iloc[-1]["close"]
        equity_curve.append({"date": df.iloc[-1]["trade_date"], "equity": final_equity})
        daily_pnl.append({"date": df.iloc[-1]["trade_date"], "pnl": final_equity - previous_equity})
        position_history.append(
            {
                "date": df.iloc[-1]["trade_date"],
                "position": int(position),
                "cash": float(cash),
                "close": float(df.iloc[-1]["close"]),
                "exposure": float(position * df.iloc[-1]["close"]),
                "weight": float((position * df.iloc[-1]["close"]) / final_equity) if final_equity > 0 else 0.0,
            }
        )

        equity_values = np.array([p["equity"] for p in equity_curve], dtype=float)
        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else np.array([])
        vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
        sharpe = (
            float((np.mean(daily_returns) / np.std(daily_returns, ddof=1)) * np.sqrt(252))
            if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 0
            else 0.0
        )

        peak = -np.inf
        max_drawdown = 0.0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - eq) / peak)

        total_return = (final_equity - self.initial_capital) / self.initial_capital
        annual_return = self._annualize_return(total_return, len(equity_curve) - 1)
        win_trades = [t for t in trades if t["action"].startswith("SELL") and t["pnl"] > 0]
        loss_trades = [t for t in trades if t["action"].startswith("SELL") and t["pnl"] <= 0]
        closed_trades = len(win_trades) + len(loss_trades)
        win_rate = len(win_trades) / closed_trades if closed_trades > 0 else 0.0
        holding_days = [
            t["holding_days"]
            for t in trades
            if t["action"].startswith("SELL") and t.get("holding_days") is not None
        ]
        avg_holding_days = float(np.mean(holding_days)) if holding_days else 0.0
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "win_rate": win_rate,
            "num_trades": closed_trades,
            "final_equity": final_equity,
            "max_drawdown": max_drawdown,
            "volatility": vol,
            "sharpe": sharpe,
            "calmar": calmar,
            "turnover": cumulative_turnover,
            "avg_holding_days": avg_holding_days,
            "equity_curve": equity_curve,
            "daily_pnl": daily_pnl,
            "position_history": position_history,
            "trades": trades,
        }

    @staticmethod
    def _annualize_return(total_return: float, periods: int) -> float:
        if periods <= 0:
            return 0.0
        growth = 1.0 + float(total_return)
        if growth <= 0:
            return -1.0
        return float(growth ** (252 / periods) - 1.0)

    @staticmethod
    def _holding_days(entry_date, exit_date) -> int | None:
        if entry_date is None or exit_date is None:
            return None
        try:
            entry = pd.to_datetime(str(entry_date), format="%Y%m%d")
            exit_ = pd.to_datetime(str(exit_date), format="%Y%m%d")
            return int((exit_ - entry).days)
        except Exception:
            return None
