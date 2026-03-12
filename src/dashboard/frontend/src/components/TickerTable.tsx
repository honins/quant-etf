import { useDeferredValue } from "react";
import type { SignalItem } from "../types/dashboard";

type TickerTableProps = {
  tickers: SignalItem[];
  filter: string;
  search: string;
  onFilterChange: (filter: string) => void;
  onSearchChange: (value: string) => void;
};

function pct(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)}%`;
}

export function TickerTable({
  tickers,
  filter,
  search,
  onFilterChange,
  onSearchChange,
}: TickerTableProps) {
  const deferredFilter = useDeferredValue(filter);
  const deferredSearch = useDeferredValue(search.trim().toLowerCase());
  const filtered = tickers.filter((item) => {
    if (deferredFilter === "all") return true;
    return item.category === deferredFilter || item.signal_bucket === deferredFilter;
  }).filter((item) => {
    if (!deferredSearch) return true;
    return (
      item.code.toLowerCase().includes(deferredSearch) ||
      item.name.toLowerCase().includes(deferredSearch) ||
      item.category_label.toLowerCase().includes(deferredSearch)
    );
  });

  const filters = [
    ["all", "全部"],
    ["buy", "买入"],
    ["watch", "观察"],
    ["core", "核心"],
    ["satellite", "卫星"],
    ["observe", "观察池"],
  ] as const;

  return (
    <section className="panel section">
      <div className="section-head">
        <div>
          <div className="section-kicker">Universe</div>
          <h2>全量标的矩阵</h2>
        </div>
        <div className="panel-note">先保留稳定筛选能力，后续再逐步增强搜索和排序。</div>
      </div>
      <div className="filter-row-modern">
        {filters.map(([value, label]) => (
          <button
            key={value}
            className={`filter-chip ${filter === value ? "active" : ""}`}
            onClick={() => onFilterChange(value)}
          >
            {label}
          </button>
        ))}
        <input
          className="search-input"
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="搜索代码 / 名称 / 分类"
        />
      </div>
      <div className="table-wrap-modern">
        <table>
          <thead>
            <tr>
              <th>代码</th>
              <th>名称</th>
              <th>分类</th>
              <th>信号</th>
              <th>模式</th>
              <th>评分</th>
              <th>阈值</th>
              <th>现价</th>
              <th>90天收益</th>
              <th>90天胜率</th>
              <th>90天交易</th>
              <th>180天收益</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item) => (
              <tr key={item.code}>
                <td>{item.code}</td>
                <td>{item.name}</td>
                <td>{item.category_label}</td>
                <td>{item.signal_bucket_label}</td>
                <td>{item.mode_label}</td>
                <td>{item.score.toFixed(4)}</td>
                <td>{item.entry_threshold.toFixed(4)}</td>
                <td>{item.current_price?.toFixed(4) ?? "-"}</td>
                <td>{pct(item.return_90d)}</td>
                <td>{pct(item.win_rate_90d)}</td>
                <td>{item.trades_90d ?? "-"}</td>
                <td>{pct(item.return_180d)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
