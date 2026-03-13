export type RecentReport = {
  name: string;
  url: string;
  updated_at: string;
  size_kb: number;
};

export type SignalRisk = {
  current_price: number | null;
  atr: number | null;
  initial_stop_loss: number | null;
  trailing_stop_loss: number | null;
  risk_per_share: number | null;
};

export type PositionSize = {
  suggested_shares: number;
  suggested_value: number;
  suggested_weight_pct: number;
};

export type SignalItem = {
  code: string;
  name: string;
  category: string;
  category_label: string;
  score: number;
  is_buy: boolean;
  market_status: string;
  market_status_label: string;
  mode: string;
  mode_label: string;
  entry_threshold: number;
  threshold_gap: number;
  current_price: number | null;
  risk: SignalRisk;
  reasons: string[];
  decision_note: string;
  position_size?: PositionSize;
  signal_bucket: string;
  signal_bucket_label: string;
  return_90d: number | null;
  win_rate_90d: number | null;
  trades_90d: number | null;
  return_180d: number | null;
  win_rate_180d: number | null;
  trades_180d: number | null;
};

export type HoldingItem = {
  code: string;
  name: string;
  buy_price: number | null;
  current_price: number | null;
  trailing_stop: number | null;
  pnl_pct: number | null;
  status: string;
  action: string;
  status_label: string;
  action_label: string;
  days_held: number;
};

export type BacktestResult = {
  code: string;
  name: string;
  category: string;
  category_label: string;
  mode: string;
  mode_label: string;
  total_return_pct: number | null;
  win_rate_pct: number | null;
  num_trades: number;
  max_drawdown_pct: number | null;
  volatility_pct: number | null;
  sharpe: number | null;
  bear_days: number;
};

export type BacktestSummary = {
  ticker_count: number;
  avg_return_pct: number | null;
  avg_max_drawdown_pct: number | null;
  avg_volatility_pct: number | null;
  positive_ratio_pct: number | null;
  overall_win_rate_pct: number | null;
  total_trades: number;
};

export type BacktestBoard = {
  window_days: number;
  start_date: string;
  end_date: string;
  start_date_label: string;
  end_date_label: string;
  summary: BacktestSummary;
  results: BacktestResult[];
  charts: Record<string, BacktestChart>;
};

export type BacktestChartPoint = {
  date: string;
  close: number | null;
  buy_price: number | null;
  sell_price: number | null;
};

export type BacktestTradePoint = {
  date: string;
  price: number | null;
  type: "buy" | "sell";
  action: string;
  pnl: number | null;
};

export type BacktestChart = {
  window_days: number;
  start_date: string;
  end_date: string;
  series: BacktestChartPoint[];
  trades: BacktestTradePoint[];
};

export type DashboardPayload = {
  generated_at: string;
  market_status: string;
  market_status_label: string;
  model_name: string;
  model_name_label: string;
  report_path: string;
  report_url: string;
  recent_reports: RecentReport[];
  controls: {
    history_days: number;
    feishu_configured: boolean;
  };
  stats: {
    active_tickers: number;
    tradable_tickers: number;
    observe_tickers: number;
    buy_count: number;
    watch_count: number;
    holdings_count: number;
    top_live_score: number | null;
    top_live_name: string | null;
    top_90_name: string | null;
    top_90_return_pct: number | null;
    top_180_name: string | null;
    top_180_return_pct: number | null;
  };
  signals: {
    buy: SignalItem[];
    watch: SignalItem[];
    observe: SignalItem[];
    all: SignalItem[];
  };
  holdings: HoldingItem[];
  histories: Record<string, Array<{ date: string; close: number | null; ma20: number | null; ma60: number | null; score: number | null }>>;
  backtests: {
    "90d": BacktestBoard;
    "180d": BacktestBoard;
  };
};

export type DashboardResponse = {
  ok: boolean;
  payload: DashboardPayload;
};

export type DashboardActionResponse = {
  ok: boolean;
  action: string;
  message: string;
  generated_at: string;
  notification_status?: string;
  notification_error?: string | null;
  payload: DashboardPayload;
};
