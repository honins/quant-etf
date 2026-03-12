import type {
  DashboardActionResponse,
  DashboardResponse,
} from "../types/dashboard";

function ensureOk<T extends { ok: boolean; error?: string }>(payload: T): T {
  if (!payload.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

export async function fetchDashboardData(historyDays: number): Promise<DashboardResponse> {
  const response = await fetch(`/api/dashboard-data?history_days=${historyDays}`, {
    cache: "no-store",
  });
  const payload = (await response.json()) as DashboardResponse & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return ensureOk(payload);
}

export async function refreshDashboard(historyDays: number): Promise<DashboardActionResponse> {
  const response = await fetch("/api/refresh-dashboard", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ history_days: historyDays }),
  });
  const payload = (await response.json()) as DashboardActionResponse & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return ensureOk(payload);
}

export async function generateReport(
  historyDays: number,
  sendNotification: boolean,
): Promise<DashboardActionResponse> {
  const response = await fetch("/api/generate-report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      history_days: historyDays,
      send_notification: sendNotification,
    }),
  });
  const payload = (await response.json()) as DashboardActionResponse & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return ensureOk(payload);
}
