from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse

from config.settings import settings
from src.core.daily_report_service import generate_daily_report
from src.dashboard.data_builder import build_dashboard_payload


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "src" / "dashboard" / "frontend" / "dist"


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


def _coerce_history_days(value: object, default: int = 120) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return default
    return max(30, min(days, 365))


def _ensure_frontend_build() -> None:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return
    raise FileNotFoundError(
        f"Frontend build not found at {index_path}. Run `npm install` and `npm run build` in `src/dashboard/frontend` first."
    )


def _read_json_request(handler: http.server.BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}

    raw = handler.rfile.read(content_length)
    if not raw:
        return {}

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


def _serve_file(handler: http.server.BaseHTTPRequestHandler, file_path: Path) -> None:
    body = file_path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(file_path))
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    if file_path.suffix in {".js", ".css"}:
        handler.send_header("Cache-Control", "public, max-age=31536000, immutable")
    else:
        handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _resolve_static_path(base_dir: Path, request_path: str) -> Path | None:
    relative = request_path.lstrip("/")
    candidate = (base_dir / relative).resolve()
    if not candidate.is_file():
        return None
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return candidate


def _resolve_frontend_path(request_path: str) -> Path | None:
    if request_path in {"/", "/dashboard.html", "/index.html"}:
        return FRONTEND_DIST_DIR / "index.html"
    return _resolve_static_path(FRONTEND_DIST_DIR, request_path)


def _resolve_report_path(request_path: str) -> Path | None:
    if request_path.startswith("/api/"):
        return None
    return _resolve_static_path(settings.REPORTS_DIR, request_path)


def _write_json(handler: http.server.BaseHTTPRequestHandler, status_code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _dashboard_payload_response(history_days: int) -> dict:
    return {
        "ok": True,
        "payload": build_dashboard_payload(history_days=history_days),
    }


def serve_dashboard(port: int, history_days: int = 120) -> None:
    _ensure_frontend_build()
    action_lock = threading.Lock()

    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            request_path = parsed.path

            if request_path == "/api/dashboard-data":
                query = parse_qs(parsed.query)
                request_history_days = _coerce_history_days(
                    query.get("history_days", [history_days])[0],
                    default=history_days,
                )
                try:
                    _write_json(self, 200, _dashboard_payload_response(request_history_days))
                except Exception as exc:
                    _write_json(self, 500, {"ok": False, "error": str(exc)})
                return

            frontend_path = _resolve_frontend_path(request_path)
            if frontend_path:
                _serve_file(self, frontend_path)
                return

            report_path = _resolve_report_path(request_path)
            if report_path:
                _serve_file(self, report_path)
                return

            self.send_error(404, "Not Found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            request_path = parsed.path
            if request_path not in {"/api/generate-report", "/api/refresh-dashboard"}:
                self.send_error(404, "Not Found")
                return

            try:
                request_payload = _read_json_request(self)
            except ValueError as exc:
                _write_json(self, 400, {"ok": False, "error": str(exc)})
                return

            request_history_days = _coerce_history_days(
                request_payload.get("history_days", history_days),
                default=history_days,
            )

            if not action_lock.acquire(blocking=False):
                _write_json(
                    self,
                    409,
                    {
                        "ok": False,
                        "error": "Another dashboard action is already running. Please wait a moment.",
                    },
                )
                return

            try:
                payload = None
                if request_path == "/api/generate-report":
                    send_notification = bool(request_payload.get("send_notification"))
                    report_result = generate_daily_report(
                        send_notification=send_notification,
                        history_days=request_history_days,
                    )
                    payload = build_dashboard_payload(history_days=request_history_days)

                    if report_result["notification_status"] == "sent":
                        message = "Today's report is ready and the Feishu notification was sent."
                    elif report_result["notification_status"] == "skipped":
                        message = "Today's report is ready. Feishu was skipped because the webhook is not configured."
                    elif report_result["notification_status"] == "failed":
                        message = "Today's report is ready, but the Feishu notification failed."
                    else:
                        message = "Today's report is ready."

                    _write_json(
                        self,
                        200,
                        {
                            "ok": True,
                            "action": "generate-report",
                            "message": message,
                            "generated_at": report_result["generated_at"],
                            "notification_status": report_result["notification_status"],
                            "notification_error": report_result["notification_error"],
                            "payload": payload,
                        },
                    )
                    return

                payload = build_dashboard_payload(history_days=request_history_days)
                _write_json(
                    self,
                    200,
                    {
                        "ok": True,
                        "action": "refresh-dashboard",
                        "message": "Dashboard refreshed. Live signals and 90/180-day backtests were recalculated.",
                        "generated_at": payload["generated_at"],
                        "payload": payload,
                    },
                )
            except Exception as exc:
                _write_json(self, 500, {"ok": False, "error": str(exc)})
            finally:
                action_lock.release()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}/dashboard.html"
    print(f"Dashboard serving at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local Quant ETF dashboard app.")
    parser.add_argument("--serve", action="store_true", help="Serve the dashboard app over HTTP.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local HTTP server.")
    parser.add_argument("--history-days", type=int, default=120, help="Ticker chart lookback window.")
    return parser.parse_args()


def main() -> None:
    _configure_stdio()
    args = parse_args()
    snapshot = build_dashboard_payload(history_days=args.history_days)
    snapshot_path = settings.REPORTS_DIR / "dashboard-data.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    print(f"Dashboard snapshot written: {snapshot_path}")
    if args.serve:
        serve_dashboard(port=args.port, history_days=args.history_days)


if __name__ == "__main__":
    main()
