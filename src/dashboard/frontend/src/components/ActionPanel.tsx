type ActionPanelProps = {
  historyDays: number;
  busy: boolean;
  status: string;
  statusTone: "neutral" | "success" | "warning" | "error";
  log: string;
  currentTask: string;
  feishuConfigured: boolean;
  onHistoryDaysChange: (days: number) => void;
  onRefresh: () => void;
  onGenerateReport: (sendNotification: boolean) => void;
};

export function ActionPanel({
  historyDays,
  busy,
  status,
  statusTone,
  log,
  currentTask,
  feishuConfigured,
  onHistoryDaysChange,
  onRefresh,
  onGenerateReport,
}: ActionPanelProps) {
  return (
    <section className="panel action-panel">
      <div className="section-head">
        <div>
          <div className="section-kicker">Actions</div>
          <h2>常用操作</h2>
        </div>
        <div className="panel-note">把高频动作收在一起，后续迭代时也更容易继续加功能。</div>
      </div>

      <div className="action-overview">
        <div className="action-overview-card">
          <span>当前任务</span>
          <strong>{busy ? currentTask : "等待操作"}</strong>
        </div>
        <div className="action-overview-card">
          <span>图表窗口</span>
          <strong>{historyDays} 天</strong>
        </div>
        <div className="action-overview-card">
          <span>飞书推送</span>
          <strong>{feishuConfigured ? "已配置" : "未配置"}</strong>
        </div>
      </div>

      <div className="history-row">
        <div>
          <div className="label">图表窗口</div>
          <div className="muted">会影响标的走势图以及工作台刷新后的计算结果。</div>
        </div>
        <select
          className="select"
          value={historyDays}
          onChange={(event) => onHistoryDaysChange(Number(event.target.value))}
        >
          {[60, 90, 120, 180, 240].map((item) => (
            <option key={item} value={item}>
              {item} 天
            </option>
          ))}
        </select>
      </div>

      <div className="button-row">
        <button className="button button-primary" disabled={busy} onClick={() => onGenerateReport(false)}>
          生成今日日报
        </button>
        <button className="button button-secondary" disabled={busy} onClick={() => onGenerateReport(true)}>
          生成并推送飞书
        </button>
        <button className="button button-ghost" disabled={busy} onClick={onRefresh}>
          刷新工作台
        </button>
      </div>

      <div className={`status-card status-${statusTone}`}>
        <div className="label">操作状态</div>
        <div className="status-text">{status}</div>
        <div className="status-log">{log}</div>
      </div>

      <div className="helper-list">
        <div>生成日报后不会强制跳转，只会更新右侧的最近产物列表。</div>
        <div>刷新会重算实时信号、持仓状态和 90/180 天回测。</div>
        <div>{feishuConfigured ? "飞书 webhook 已配置，可以直接推送。" : "飞书 webhook 未配置，推送动作会自动跳过发送。"}</div>
      </div>
    </section>
  );
}
