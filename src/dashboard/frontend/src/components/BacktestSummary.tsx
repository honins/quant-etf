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

function num(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined ? "-" : value.toFixed(digits);
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
  const champion = payload.benchmark.champion;
  const challenger = payload.benchmark.challenger;

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Backtests</div>
          <h2>回测摘要</h2>
        </div>
        <div className="panel-note">保留 90/180 天视角，但用更稳定的卡片与图表布局。</div>
      </div>
      <div className="benchmark-banner">
        <div className="benchmark-card champion">
          <span>Champion</span>
          <strong>{champion.model_name ?? "-"}</strong>
          <small>{champion.validation_mode ?? "-"} / Sharpe {num(champion.avg_sharpe)}</small>
        </div>
        <div className="benchmark-card challenger">
          <span>Challenger</span>
          <strong>{challenger.model_name ?? "-"}</strong>
          <small>{challenger.validation_mode ?? "-"} / Sharpe {num(challenger.avg_sharpe)}</small>
        </div>
      </div>
      <div className="benchmark-history-strip">
        {(payload.benchmark.history ?? []).slice(0, 6).map((item, index) => (
          <span key={`${item.model_name ?? "m"}-${item.validation_mode ?? "v"}-${index}`}>
            {item.model_name ?? "-"}/{item.validation_mode ?? "-"} {num(item.avg_sharpe)}
          </span>
        ))}
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
              <SummaryCard
                title="年化 / Sharpe"
                value={`${pct(board.summary.avg_annual_return_pct ?? null)} / ${num(board.summary.avg_sharpe)}`}
                subtitle={`Calmar ${num(board.summary.avg_calmar)}`}
              />
              <SummaryCard
                title="换手 / 持有"
                value={`${pct(board.summary.avg_turnover_pct ?? null)} / ${num(board.summary.avg_holding_days, 1)}天`}
                subtitle={`组合建议 ${board.portfolio.top_k.length} 只`}
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
            <div className="portfolio-plan">
              <div className="portfolio-plan-head">
                <strong>组合建议</strong>
                <span>预期换手 {pct(board.portfolio.expected_turnover_pct)}</span>
              </div>
              <div className="portfolio-meta-row">
                <span>平均相关性 {num(Number(board.portfolio.diagnostics?.avg_correlation ?? 0), 3)}</span>
                <span>平均成本 {pct(Number(board.portfolio.diagnostics?.avg_trading_cost ?? 0) * 100)}</span>
              </div>
              <div className="portfolio-pill-row">
                {board.portfolio.top_k.map((code) => (
                  <span key={code} className="portfolio-pill">{code}</span>
                ))}
              </div>
              <div className="portfolio-list">
                {board.portfolio.recommendations.slice(0, 5).map((item) => (
                  <div key={item.code} className="portfolio-row">
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.code}</span>
                    </div>
                    <div>
                      <strong>{pct(item.suggested_weight_pct)}</strong>
                      <span>权重</span>
                    </div>
                    <div>
                      <strong>{pct(item.expected_return_pct)}</strong>
                      <span>预期收益</span>
                    </div>
                    <div>
                      <strong>{num(item.confidence, 3)}</strong>
                      <span>信心</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="allocation-replay">
                <div className="portfolio-plan-head">
                  <strong>组合回放</strong>
                  <span>组合 ROI {pct(board.portfolio_backtest.portfolio_roi_pct)}</span>
                </div>
                <div className="portfolio-meta-row">
                  <span>波动 {pct(board.portfolio_backtest.portfolio_volatility_pct)}</span>
                  <span>天数 {board.portfolio_backtest.portfolio_num_days}</span>
                  <span>最大回撤 {pct(board.portfolio_backtest.portfolio_max_drawdown_pct)}</span>
                  <span>期末净值 {num(board.portfolio_backtest.portfolio_ending_equity, 3)}</span>
                  <span>上涨天数 {board.portfolio_backtest.portfolio_positive_days}</span>
                  <span>下跌天数 {board.portfolio_backtest.portfolio_negative_days}</span>
                </div>
                <div className="variant-strip">
                  {Object.entries(board.portfolio_variants ?? {}).map(([name, variant]) => (
                    <span key={name}>
                      {name}: ROI {pct(variant.roi_pct)} / DD {pct(variant.max_drawdown_pct)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
