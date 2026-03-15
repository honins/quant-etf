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
    <button
      type="button"
      className={`signal-card signal-${tone} ${selected ? "signal-selected" : ""}`}
      onClick={onSelect}
    >
      <div className="signal-header">
        <div>
          <div className="signal-title">{item.name}</div>
          <div className="signal-subtitle">
            {item.code} / {item.category_label} / {item.mode_label}
          </div>
        </div>
        <div className="signal-score">
          <div>{item.score.toFixed(4)}</div>
          <span>
            {tone === "buy"
              ? `\u9608\u503c ${item.entry_threshold.toFixed(4)}`
              : item.signal_bucket_label}
          </span>
        </div>
      </div>
      <div className="signal-metrics">
        <div>
          <span>{"\u73b0\u4ef7"}</span>
          <strong>{fixed(item.current_price)}</strong>
        </div>
        <div>
          <span>{"\u521d\u59cb\u6b62\u635f"}</span>
          <strong>{fixed(item.risk.initial_stop_loss)}</strong>
        </div>
        <div>
          <span>{"\u8ddf\u8e2a\u6b62\u635f"}</span>
          <strong>{fixed(item.risk.trailing_stop_loss)}</strong>
        </div>
        <div>
          <span>{"\u8fd190\u5929\u6536\u76ca"}</span>
          <strong>{pct(item.return_90d)}</strong>
        </div>
      </div>
      {item.reasons.length > 0 ? (
        <div className="signal-reason-row">
          {item.reasons.slice(0, 2).map((reason) => (
            <span key={reason} className="signal-reason-chip" title={reason}>
              {reason}
            </span>
          ))}
        </div>
      ) : null}
    </button>
  );
}

export function SignalBoard({
  buy,
  watch,
  observe,
  selectedCode,
  onSelectTicker,
}: SignalBoardProps) {
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
          <h2>{"\u5b9e\u65f6\u4fe1\u53f7\u961f\u5217"}</h2>
        </div>
        <div className="panel-note">
          {`\u5171 ${merged.length} \u5f20\u5361\u7247\uff0c\u70b9\u51fb\u53ef\u5207\u6362\u53f3\u4fa7\u8be6\u60c5`}
        </div>
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
          <div className="empty-card">
            {"\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u5b9e\u65f6\u4fe1\u53f7\u3002"}
          </div>
        )}
      </div>
    </section>
  );
}
