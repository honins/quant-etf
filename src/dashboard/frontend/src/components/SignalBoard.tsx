import type { SignalItem } from "../types/dashboard";

type SignalBoardProps = {
  buy: SignalItem[];
  watch: SignalItem[];
  observe: SignalItem[];
  selectedCode: string | null;
  onSelectTicker: (code: string) => void;
};

function pct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

function fixed(value: number | null, digits = 4) {
  return value === null ? "-" : value.toFixed(digits);
}

function SignalCard({
  item,
  tone,
  selected,
  onSelect,
}: {
  item: SignalItem;
  tone: "buy" | "watch" | "observe";
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`signal-card signal-${tone} ${selected ? "signal-selected" : ""}`} onClick={onSelect}>
      <div className="signal-header">
        <div>
          <div className="signal-title">{item.name}</div>
          <div className="signal-subtitle">
            {item.code} / {item.category_label} / {item.mode_label}
          </div>
        </div>
        <div className="signal-score">
          <div>{item.score.toFixed(4)}</div>
          <span>{tone === "buy" ? `阈值 ${item.entry_threshold.toFixed(4)}` : item.signal_bucket_label}</span>
        </div>
      </div>
      <div className="signal-metrics">
        <div>
          <span>现价</span>
          <strong>{fixed(item.current_price)}</strong>
        </div>
        <div>
          <span>初始止损</span>
          <strong>{fixed(item.risk.initial_stop_loss)}</strong>
        </div>
        <div>
          <span>跟踪止损</span>
          <strong>{fixed(item.risk.trailing_stop_loss)}</strong>
        </div>
        <div>
          <span>近90天收益</span>
          <strong>{pct(item.return_90d)}</strong>
        </div>
      </div>
      {item.reasons.length > 0 ? (
        <ul className="reason-list">
          {item.reasons.slice(0, 3).map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
    </button>
  );
}

export function SignalBoard({ buy, watch, observe, selectedCode, onSelectTicker }: SignalBoardProps) {
  const merged = [
    ...buy.map((item) => ({ item, tone: "buy" as const })),
    ...watch.slice(0, 4).map((item) => ({ item, tone: "watch" as const })),
    ...observe.slice(0, 2).map((item) => ({ item, tone: "observe" as const })),
  ];

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Signals</div>
          <h2>实时信号队列</h2>
        </div>
        <div className="panel-note">点击卡片即可切换右侧的标的详情。</div>
      </div>
      <div className="signal-list">
        {merged.length ? (
          merged.map(({ item, tone }) => (
            <SignalCard
              key={item.code}
              item={item}
              tone={tone}
              selected={item.code === selectedCode}
              onSelect={() => onSelectTicker(item.code)}
            />
          ))
        ) : (
          <div className="empty-card">当前没有可展示的实时信号。</div>
        )}
      </div>
    </section>
  );
}
