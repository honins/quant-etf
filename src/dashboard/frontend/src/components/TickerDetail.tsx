import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BacktestChart, DashboardPayload } from "../types/dashboard";

type TickerDetailProps = {
  payload: DashboardPayload;
  selectedCode: string | null;
  onSelectTicker: (code: string) => void;
};

type HistoryPoint = DashboardPayload["histories"][string][number];
type ExpandedChart = "history" | "backtest" | null;

function pct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

function fixed(value: number | null) {
  return value === null ? "-" : value.toFixed(4);
}

function Metric({ title, value }: { title: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  compact = false,
  expandDisabled = false,
  onExpand,
  children,
}: {
  title: string;
  subtitle?: string;
  compact?: boolean;
  expandDisabled?: boolean;
  onExpand?: () => void;
  children: ReactNode;
}) {
  return (
    <div className={`chart-shell${compact ? " compact" : ""}`}>
      <div className="chart-toolbar">
        <div className="chart-copy">
          <div className="chart-title">{title}</div>
          {subtitle ? <div className="chart-caption">{subtitle}</div> : null}
        </div>
        {onExpand ? (
          <button
            type="button"
            className="button button-ghost chart-action"
            onClick={onExpand}
            disabled={expandDisabled}
          >
            {"\u653e\u5927\u67e5\u770b"}
          </button>
        ) : null}
      </div>
      {children}
    </div>
  );
}

function HistoryTrendChart({
  history,
  height,
}: {
  history: HistoryPoint[];
  height: number | string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={history}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(87, 96, 113, 0.18)" />
        <XAxis dataKey="date" minTickGap={28} stroke="#6b7280" />
        <YAxis yAxisId="price" stroke="#6b7280" />
        <YAxis yAxisId="score" orientation="right" stroke="#6b7280" domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          name={"\u6536\u76d8\u4ef7"}
          stroke="#1f4a57"
          dot={false}
          strokeWidth={2.5}
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="ma20"
          name="MA20"
          stroke="#d59a2f"
          dot={false}
          strokeWidth={1.8}
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="ma60"
          name="MA60"
          stroke="#67a58c"
          dot={false}
          strokeWidth={1.8}
        />
        <Line
          yAxisId="score"
          type="monotone"
          dataKey="score"
          name={"AI\u8bc4\u5206"}
          stroke="#bc5c40"
          dot={false}
          strokeWidth={2}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function BacktestTradeChart({
  chart,
  height,
}: {
  chart: BacktestChart;
  height: number | string;
}) {
  const buyDots = chart.series
    .filter((item) => item.buy_price !== null)
    .map((item) => ({ date: item.date, price: item.buy_price as number }));
  const sellDots = chart.series
    .filter((item) => item.sell_price !== null)
    .map((item) => ({ date: item.date, price: item.sell_price as number }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chart.series}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(87, 96, 113, 0.18)" />
        <XAxis dataKey="date" minTickGap={28} stroke="#6b7280" />
        <YAxis stroke="#6b7280" />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="close"
          name={"\u56de\u6d4b\u6536\u76d8\u4ef7"}
          stroke="#1f4a57"
          dot={false}
          strokeWidth={2.2}
        />
        <Scatter
          data={buyDots}
          name={"\u4e70\u70b9"}
          dataKey="price"
          fill="#1b8a67"
        />
        <Scatter
          data={sellDots}
          name={"\u5356\u70b9"}
          dataKey="price"
          fill="#c35b3d"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function TickerDetail({ payload, selectedCode, onSelectTicker }: TickerDetailProps) {
  const [backtestWindow, setBacktestWindow] = useState<"90d" | "180d">("90d");
  const [expandedChart, setExpandedChart] = useState<ExpandedChart>(null);
  const selected =
    payload.signals.all.find((item) => item.code === selectedCode) ??
    payload.signals.buy[0] ??
    payload.signals.all[0] ??
    null;

  const history = selected ? payload.histories[selected.code] || [] : [];
  const selectedBacktestChart = selected
    ? payload.backtests[backtestWindow].charts[selected.code]
    : undefined;
  const selectedLabel = selected ? `${selected.name} (${selected.code})` : "";
  const modalTitle =
    expandedChart === "history" ? "\u8d70\u52bf\u56fe\u5927\u56fe" : "\u56de\u6d4b\u4e70\u5356\u70b9\u5927\u56fe";
  const modalSubtitle =
    expandedChart === "backtest"
      ? `${selectedLabel} / ${backtestWindow === "90d" ? "Recent 90D" : "Recent 180D"}`
      : selectedLabel;

  useEffect(() => {
    if (!expandedChart) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setExpandedChart(null);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [expandedChart]);

  useEffect(() => {
    if (expandedChart === "history" && history.length === 0) {
      setExpandedChart(null);
    }
    if (expandedChart === "backtest" && !selectedBacktestChart) {
      setExpandedChart(null);
    }
  }, [expandedChart, history.length, selectedBacktestChart]);

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Ticker</div>
          <h2>{"\u6807\u7684\u8be6\u60c5"}</h2>
        </div>
        <select
          className="select"
          value={selected?.code ?? ""}
          onChange={(event) => onSelectTicker(event.target.value)}
        >
          {payload.signals.all.map((item) => (
            <option key={item.code} value={item.code}>
              {item.name} ({item.code})
            </option>
          ))}
        </select>
      </div>
      {selected ? (
        <>
          <div className="ticker-spotlight">
            <div className="ticker-spotlight-head">
              <div>
                <div className="signal-title">{selected.name}</div>
                <div className="signal-subtitle">
                  {selected.code} / {selected.category_label} / {selected.mode_label}
                </div>
              </div>
              <div className="ticker-score-badge">{selected.score.toFixed(4)}</div>
            </div>
            <div className="ticker-summary">
              <Metric title={"\u73b0\u4ef7"} value={fixed(selected.current_price)} />
              <Metric title={"\u521d\u59cb\u6b62\u635f"} value={fixed(selected.risk.initial_stop_loss)} />
              <Metric title={"\u8ddf\u8e2a\u6b62\u635f"} value={fixed(selected.risk.trailing_stop_loss)} />
              <Metric
                title={"\u5efa\u8bae\u4ed3\u4f4d"}
                value={
                  selected.position_size
                    ? `${selected.position_size.suggested_weight_pct.toFixed(2)}%`
                    : "-"
                }
              />
              <Metric title={"\u8fd190\u5929\u6536\u76ca"} value={pct(selected.return_90d)} />
              <Metric title={"\u8fd1180\u5929\u6536\u76ca"} value={pct(selected.return_180d)} />
            </div>
            {selected.reasons.length ? (
              <div className="insight-list">
                {selected.reasons.slice(0, 4).map((reason) => (
                  <div key={reason} className="insight-chip">
                    {reason}
                  </div>
                ))}
              </div>
            ) : null}
            {selected.decision_note ? (
              <div className="decision-note">{selected.decision_note}</div>
            ) : null}
          </div>
          <div className="ticker-summary">
            <Metric title={"\u6700\u65b0\u8bc4\u5206"} value={selected.score.toFixed(4)} />
            <Metric title={"\u5165\u573a\u9608\u503c"} value={selected.entry_threshold.toFixed(4)} />
            <Metric title={"\u521d\u59cb\u6b62\u635f"} value={fixed(selected.risk.initial_stop_loss)} />
            <Metric title={"\u8ddf\u8e2a\u6b62\u635f"} value={fixed(selected.risk.trailing_stop_loss)} />
            <Metric title={"\u8fd190\u5929\u6536\u76ca"} value={pct(selected.return_90d)} />
            <Metric title={"\u8fd1180\u5929\u6536\u76ca"} value={pct(selected.return_180d)} />
          </div>

          <ChartCard
            title={"\u8d70\u52bf\u56fe"}
            subtitle={selectedLabel}
            onExpand={() => setExpandedChart("history")}
            expandDisabled={history.length === 0}
          >
            <HistoryTrendChart history={history} height={320} />
          </ChartCard>

          <div className="section-head" style={{ marginTop: 14 }}>
            <div>
              <div className="section-kicker">Backtest</div>
              <h2>{"\u56de\u6d4b\u4e70\u5356\u70b9"}</h2>
            </div>
            <select
              className="select"
              value={backtestWindow}
              onChange={(event) => setBacktestWindow(event.target.value as "90d" | "180d")}
            >
              <option value="90d">90 D</option>
              <option value="180d">180 D</option>
            </select>
          </div>
          {selectedBacktestChart ? (
            <ChartCard
              title={"\u56de\u6d4b\u4e70\u5356\u70b9"}
              subtitle={backtestWindow === "90d" ? "Recent 90D" : "Recent 180D"}
              compact
              onExpand={() => setExpandedChart("backtest")}
            >
              <BacktestTradeChart chart={selectedBacktestChart} height={280} />
            </ChartCard>
          ) : (
            <div className="empty-card">
              {"\u8be5\u6807\u7684\u5f53\u524d\u6ca1\u6709\u56de\u6d4b\u4e70\u5356\u70b9\u6570\u636e\u3002"}
            </div>
          )}

          {expandedChart && typeof document !== "undefined"
            ? createPortal(
                <div
                  className="chart-modal-overlay"
                  onClick={() => setExpandedChart(null)}
                  role="presentation"
                >
                  <div
                    className="chart-modal-card panel"
                    role="dialog"
                    aria-modal="true"
                    aria-label={modalTitle}
                    onClick={(event) => event.stopPropagation()}
                  >
                    <div className="chart-modal-head">
                      <div>
                        <div className="section-kicker">
                          {expandedChart === "history" ? "Trend" : "Backtest"}
                        </div>
                        <h3>{modalTitle}</h3>
                        <div className="chart-caption">{modalSubtitle}</div>
                      </div>
                      <button
                        type="button"
                        className="button button-secondary chart-action"
                        onClick={() => setExpandedChart(null)}
                      >
                        {"\u5173\u95ed"}
                      </button>
                    </div>
                    <div className="chart-shell chart-shell-modal">
                      <div className="chart-modal-canvas">
                        {expandedChart === "history" ? (
                          <HistoryTrendChart history={history} height="100%" />
                        ) : selectedBacktestChart ? (
                          <BacktestTradeChart chart={selectedBacktestChart} height="100%" />
                        ) : null}
                      </div>
                    </div>
                  </div>
                </div>,
                document.body,
              )
            : null}
        </>
      ) : (
        <div className="empty-card">
          {"\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u6807\u7684\u3002"}
        </div>
      )}
    </section>
  );
}
