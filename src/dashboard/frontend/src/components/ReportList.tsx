import type { RecentReport } from "../types/dashboard";

type ReportListProps = {
  reportPath: string;
  reportUrl: string;
  reports: RecentReport[];
};

export function ReportList({ reportPath, reportUrl, reports }: ReportListProps) {
  return (
    <section className="panel report-panel">
      <div className="section-head">
        <div>
          <div className="section-kicker">Reports</div>
          <h2>最近产物</h2>
        </div>
        <a className="report-open-link" href={reportUrl} target="_blank" rel="noreferrer">
          打开今日日报
        </a>
      </div>
      <div className="muted report-current">{reportPath}</div>
      <div className="report-list">
        {reports.length ? (
          reports.map((report) => (
            <a key={report.name} className="report-item" href={report.url} target="_blank" rel="noreferrer">
              <div className="report-item-name">{report.name}</div>
              <div className="report-item-meta">
                {report.updated_at} · {report.size_kb} KB
              </div>
            </a>
          ))
        ) : (
          <div className="empty-card">当前还没有日报文件，可以先用左侧按钮生成一份。</div>
        )}
      </div>
    </section>
  );
}
