from __future__ import annotations

import argparse
import functools
import http.server
import json
from pathlib import Path
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.settings import settings
from src.dashboard.data_builder import build_dashboard_payload


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


def render_dashboard(output_path: Path, history_days: int = 120) -> Path:
    payload = build_dashboard_payload(history_days=history_days)
    template_dir = Path(__file__).resolve().parent / "src" / "dashboard" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(("html", "xml")),
    )
    template = env.get_template("dashboard.html.j2")
    html = template.render(
        generated_at=payload["generated_at"],
        market_status=payload["market_status"],
        model_name=payload["model_name"],
        report_path=payload["report_path"],
        stats=payload["stats"],
        signals=payload["signals"],
        holdings=payload["holdings"],
        backtests=payload["backtests"],
        payload_json=json.dumps(payload, ensure_ascii=False),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def serve_dashboard(directory: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/dashboard.html"
    print(f"Dashboard serving at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the local Quant ETF dashboard.")
    parser.add_argument("--serve", action="store_true", help="Serve reports/dashboard.html over HTTP.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local HTTP server.")
    parser.add_argument("--history-days", type=int, default=120, help="Ticker chart lookback window.")
    return parser.parse_args()


def main() -> None:
    _configure_stdio()
    args = parse_args()
    output_path = settings.REPORTS_DIR / "dashboard.html"
    rendered = render_dashboard(output_path, history_days=args.history_days)
    print(f"Dashboard generated: {rendered}")
    if args.serve:
        serve_dashboard(rendered.parent, port=args.port)


if __name__ == "__main__":
    main()
