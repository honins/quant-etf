import type { HoldingItem } from "../types/dashboard";

type HoldingsPanelProps = {
  holdings: HoldingItem[];
};

function pct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

function fixed(value: number | null) {
  return value === null ? "-" : value.toFixed(4);
}

export function HoldingsPanel({ holdings }: HoldingsPanelProps) {
  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Holdings</div>
          <h2>持仓监控</h2>
        </div>
        <div className="panel-note">跟踪浮盈亏、跟踪止损和建议动作。</div>
      </div>
      <div className="holding-list">
        {holdings.length ? (
          holdings.map((item) => (
            <article key={item.code} className="holding-card">
              <div className="holding-head">
                <div>
                  <div className="holding-title">{item.name}</div>
                  <div className="holding-subtitle">
                    {item.code} / 持有 {item.days_held} 天
                  </div>
                </div>
                <div className={`holding-pnl ${item.pnl_pct !== null && item.pnl_pct >= 0 ? "tone-positive" : "tone-negative"}`}>
                  {pct(item.pnl_pct)}
                </div>
              </div>
              <div className="holding-grid">
                <div>
                  <span>成本价</span>
                  <strong>{fixed(item.buy_price)}</strong>
                </div>
                <div>
                  <span>最新价</span>
                  <strong>{fixed(item.current_price)}</strong>
                </div>
                <div>
                  <span>跟踪止损</span>
                  <strong>{fixed(item.trailing_stop)}</strong>
                </div>
                <div>
                  <span>建议动作</span>
                  <strong>{item.action_label}</strong>
                </div>
              </div>
            </article>
          ))
        ) : (
          <div className="empty-card">当前没有持仓记录。</div>
        )}
      </div>
    </section>
  );
}
