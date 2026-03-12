import type { DashboardPayload } from "../types/dashboard";

type TopBarProps = {
  payload: DashboardPayload;
};

export function TopBar({ payload }: TopBarProps) {
  return (
    <section className="hero-card">
      <div className="eyebrow">Quant ETF Workbench</div>
      <h1>更现代的单页工作台，把常用动作和关键决策放在同一页。</h1>
      <p className="hero-copy">
        保留现在的数据逻辑，但把页面切成更容易迭代的前端应用。日报、刷新、持仓跟踪、回测摘要和标的细节都放在一个稳定结构里。
      </p>
      <div className="hero-pills">
        <span className="pill">市场: {payload.market_status_label}</span>
        <span className="pill">模型: {payload.model_name_label}</span>
        <span className="pill">更新时间: {payload.generated_at}</span>
        <span className="pill">图表窗口: {payload.controls.history_days} 天</span>
      </div>
    </section>
  );
}
