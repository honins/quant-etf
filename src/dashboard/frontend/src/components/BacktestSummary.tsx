import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardPayload } from "../types/dashboard";

type BacktestSummaryProps = {
  payload: DashboardPayload;
};

function pct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

function SummaryCard({
  title,
  value,
  subtitle,
}: {
  title: string;
  value: string;
  subtitle: string;
}) {
  return (
    <div className="summary-card-modern">
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{subtitle}</small>
    </div>
  );
}

export function BacktestSummary({ payload }: BacktestSummaryProps) {
  const boards = [payload.backtests["90d"], payload.backtests["180d"]];

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Backtests</div>
          <h2>回测摘要</h2>
        </div>
        <div className="panel-note">保留 90/180 天视角，但用更稳定的卡片与图表布局。</div>
      </div>
      <div className="backtest-grid">
        {boards.map((board) => (
          <div key={board.window_days} className="backtest-board">
            <div className="backtest-board-head">
              <h3>{board.window_days} 天回测</h3>
              <span>
                {board.start_date_label} - {board.end_date_label}
              </span>
            </div>
            <div className="summary-grid-modern">
              <SummaryCard
                title="平均收益"
                value={pct(board.summary.avg_return_pct)}
                subtitle={`胜率 ${pct(board.summary.overall_win_rate_pct)}`}
              />
              <SummaryCard
                title="平均回撤"
                value={pct(board.summary.avg_max_drawdown_pct)}
                subtitle={`交易 ${board.summary.total_trades}`}
              />
              <SummaryCard
                title="正收益占比"
                value={pct(board.summary.positive_ratio_pct)}
                subtitle={`样本 ${board.summary.ticker_count}`}
              />
            </div>
            <div className="chart-shell compact">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={board.results.slice(0, 8)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(87, 96, 113, 0.18)" />
                  <XAxis dataKey="name" stroke="#6b7280" interval={0} angle={-18} textAnchor="end" height={60} />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Bar dataKey="total_return_pct" fill="#214754" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
