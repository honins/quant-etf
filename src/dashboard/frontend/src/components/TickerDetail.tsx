import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardPayload } from "../types/dashboard";

type TickerDetailProps = {
  payload: DashboardPayload;
  selectedCode: string | null;
  onSelectTicker: (code: string) => void;
};

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

export function TickerDetail({ payload, selectedCode, onSelectTicker }: TickerDetailProps) {
  const selected =
    payload.signals.all.find((item) => item.code === selectedCode) ??
    payload.signals.buy[0] ??
    payload.signals.all[0] ??
    null;

  const history = selected ? payload.histories[selected.code] || [] : [];

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Ticker</div>
          <h2>标的详情</h2>
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
              <Metric title="现价" value={fixed(selected.current_price)} />
              <Metric title="初始止损" value={fixed(selected.risk.initial_stop_loss)} />
              <Metric title="跟踪止损" value={fixed(selected.risk.trailing_stop_loss)} />
              <Metric
                title="建议仓位"
                value={
                  selected.position_size
                    ? `${selected.position_size.suggested_weight_pct.toFixed(2)}%`
                    : "-"
                }
              />
              <Metric title="近90天收益" value={pct(selected.return_90d)} />
              <Metric title="近180天收益" value={pct(selected.return_180d)} />
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
            {selected.decision_note ? <div className="decision-note">{selected.decision_note}</div> : null}
          </div>
          <div className="ticker-summary">
            <Metric title="最新评分" value={selected.score.toFixed(4)} />
            <Metric title="入场阈值" value={selected.entry_threshold.toFixed(4)} />
            <Metric title="初始止损" value={fixed(selected.risk.initial_stop_loss)} />
            <Metric title="跟踪止损" value={fixed(selected.risk.trailing_stop_loss)} />
            <Metric title="近90天收益" value={pct(selected.return_90d)} />
            <Metric title="近180天收益" value={pct(selected.return_180d)} />
          </div>
          <div className="chart-shell">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(87, 96, 113, 0.18)" />
                <XAxis dataKey="date" minTickGap={28} stroke="#6b7280" />
                <YAxis yAxisId="price" stroke="#6b7280" />
                <YAxis yAxisId="score" orientation="right" stroke="#6b7280" domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line yAxisId="price" type="monotone" dataKey="close" name="收盘价" stroke="#1f4a57" dot={false} strokeWidth={2.5} />
                <Line yAxisId="price" type="monotone" dataKey="ma20" name="MA20" stroke="#d59a2f" dot={false} strokeWidth={1.8} />
                <Line yAxisId="price" type="monotone" dataKey="ma60" name="MA60" stroke="#67a58c" dot={false} strokeWidth={1.8} />
                <Line yAxisId="score" type="monotone" dataKey="score" name="AI评分" stroke="#bc5c40" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="empty-card">当前没有可展示的标的。</div>
      )}
    </section>
  );
}
