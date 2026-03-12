import { startTransition, useEffect, useRef, useState } from "react";
import { fetchDashboardData, generateReport, refreshDashboard } from "./api/dashboard";
import { ActionPanel } from "./components/ActionPanel";
import { BacktestSummary } from "./components/BacktestSummary";
import { HoldingsPanel } from "./components/HoldingsPanel";
import { ReportList } from "./components/ReportList";
import { SignalBoard } from "./components/SignalBoard";
import { TickerDetail } from "./components/TickerDetail";
import { TickerTable } from "./components/TickerTable";
import { TopBar } from "./components/TopBar";
import type { DashboardPayload } from "./types/dashboard";

type StatusTone = "neutral" | "success" | "warning" | "error";

function formatPct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

function StatCard({ title, value, subtitle }: { title: string; value: string; subtitle: string }) {
  return (
    <section className="panel stat-card">
      <div className="label">{title}</div>
      <div className="stat-value">{value}</div>
      <div className="muted">{subtitle}</div>
    </section>
  );
}

export default function App() {
  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [historyDays, setHistoryDays] = useState(120);
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("正在加载工作台数据...");
  const [statusTone, setStatusTone] = useState<StatusTone>("neutral");
  const [log, setLog] = useState("准备初始化...");
  const [currentTask, setCurrentTask] = useState("初始化工作台");
  const initializedRef = useRef(false);

  useEffect(() => {
    void loadDashboard(120);
  }, []);

  useEffect(() => {
    if (!initializedRef.current) {
      return;
    }
    void loadDashboard(historyDays);
  }, [historyDays]);

  async function loadDashboard(days: number) {
    setBusy(true);
    setCurrentTask("加载工作台");
    setStatus("正在加载工作台数据...");
    setStatusTone("warning");
    try {
      const response = await fetchDashboardData(days);
      startTransition(() => {
        setPayload(response.payload);
        setHistoryDays(response.payload.controls.history_days);
        setSelectedCode((current) =>
          current && response.payload.signals.all.some((item) => item.code === current)
            ? current
            : response.payload.signals.buy[0]?.code ?? response.payload.signals.all[0]?.code ?? null,
        );
      });
      initializedRef.current = true;
      setStatus("工作台已就绪，可以开始操作。");
      setStatusTone("success");
      setLog(`最近一次刷新: ${response.payload.generated_at}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "加载失败");
      setStatusTone("error");
      setLog(`失败时间: ${new Date().toLocaleString()}`);
    } finally {
      setBusy(false);
      setCurrentTask("等待操作");
    }
  }

  async function handleRefresh() {
    setBusy(true);
    setCurrentTask("刷新工作台");
    setStatus("正在刷新工作台数据...");
    setStatusTone("warning");
    try {
      const response = await refreshDashboard(historyDays);
      startTransition(() => {
        setPayload(response.payload);
      });
      setStatus(response.message);
      setStatusTone("success");
      setLog(`最近一次操作: ${response.generated_at}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "刷新失败");
      setStatusTone("error");
      setLog(`失败时间: ${new Date().toLocaleString()}`);
    } finally {
      setBusy(false);
      setCurrentTask("等待操作");
    }
  }

  async function handleGenerateReport(sendNotification: boolean) {
    setBusy(true);
    setCurrentTask(sendNotification ? "生成并推送日报" : "生成今日日报");
    setStatus(sendNotification ? "正在生成日报并尝试推送飞书..." : "正在生成今日日报...");
    setStatusTone("warning");
    try {
      const response = await generateReport(historyDays, sendNotification);
      startTransition(() => {
        setPayload(response.payload);
      });
      setStatus(response.message);
      setStatusTone(response.notification_status === "failed" ? "error" : "success");
      setLog(`最近一次操作: ${response.generated_at}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "生成日报失败");
      setStatusTone("error");
      setLog(`失败时间: ${new Date().toLocaleString()}`);
    } finally {
      setBusy(false);
      setCurrentTask("等待操作");
    }
  }

  if (!payload) {
    return (
      <main className="app-shell loading-shell">
        <section className="panel loading-card">
          <div className="section-kicker">Loading</div>
          <h2>正在准备 Quant ETF 工作台...</h2>
          <div className="muted">{status}</div>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <TopBar payload={payload} />

      <section className="top-grid">
        <ActionPanel
          historyDays={historyDays}
          busy={busy}
          status={status}
          statusTone={statusTone}
          log={log}
          currentTask={currentTask}
          feishuConfigured={payload.controls.feishu_configured}
          onHistoryDaysChange={setHistoryDays}
          onRefresh={handleRefresh}
          onGenerateReport={handleGenerateReport}
        />
        <ReportList reportPath={payload.report_path} reportUrl={payload.report_url} reports={payload.recent_reports} />
      </section>

      <section className="kpi-grid-modern">
        <StatCard title="活跃标的" value={String(payload.stats.active_tickers)} subtitle={`可交易 ${payload.stats.tradable_tickers} / 观察 ${payload.stats.observe_tickers}`} />
        <StatCard title="买入信号" value={String(payload.stats.buy_count)} subtitle={`关注名单 ${payload.stats.watch_count}`} />
        <StatCard title="当前持仓" value={String(payload.stats.holdings_count)} subtitle="来自 holdings.yml" />
        <StatCard title="90天平均收益" value={formatPct(payload.backtests["90d"].summary.avg_return_pct)} subtitle={`胜率 ${formatPct(payload.backtests["90d"].summary.overall_win_rate_pct)}`} />
        <StatCard title="180天平均收益" value={formatPct(payload.backtests["180d"].summary.avg_return_pct)} subtitle={`胜率 ${formatPct(payload.backtests["180d"].summary.overall_win_rate_pct)}`} />
        <StatCard title="最佳标的" value={payload.stats.top_live_name ?? "-"} subtitle={`最高评分 ${payload.stats.top_live_score?.toFixed(4) ?? "-"}`} />
      </section>

      <section className="main-grid">
        <div className="stack">
          <SignalBoard
            buy={payload.signals.buy}
            watch={payload.signals.watch}
            observe={payload.signals.observe}
            selectedCode={selectedCode}
            onSelectTicker={setSelectedCode}
          />
        </div>
        <div className="stack">
          <TickerDetail payload={payload} selectedCode={selectedCode} onSelectTicker={setSelectedCode} />
          <HoldingsPanel holdings={payload.holdings} />
        </div>
      </section>

      <BacktestSummary payload={payload} />
      <TickerTable
        tickers={payload.signals.all}
        filter={filter}
        search={search}
        onFilterChange={setFilter}
        onSearchChange={setSearch}
      />
    </main>
  );
}
